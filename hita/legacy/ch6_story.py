"""Chapter 6 — frequentist sampling variability on the Ch4 duo (2D | 3D) stage."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch

from ch5_core import (
    CH5_BELIEF_SURFACE_ALPHA,
    CH5_BEST_LINE_N_LAYOUT,
    CH5_VIEW_BOUNDS,
    ch5_hq_land_elev,
    ch5_plot_belief_surface_with_grid,
)
from ch4_layout import ch4_nll_heatmap_facecolors
from ch5_datasets import ch5_plot_limits, ch5_unpack_dataset
from ch5_layout import (
    ch5_composite_2x2_quadrants,
    ch5_crossfade_images,
    ch5_draw_hpd_voxels,
    ch5_uniform_belief_facecolors,
    ch5_uniform_belief_rgba_at_pdf,
    ch5_uniform_belief_z_lim,
)
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
    CH6_WILD_COV_FAIL,
    CH6_WILD_COV_PASS,
    CH6_WILD_MU_FAIL,
    CH6_WILD_MU_PASS,
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
    ch6_nll,
    ch6_nll_grad,
    ch6_observed_information,
    ch6_gaussian_ellipse_points,
    ch6_hessian_eigen_w12,
    ch6_iso_vs_corr_clouds,
    ch6_landing_histogram,
    ch6_lr_grid_w12,
    ch6_lr_threshold,
    ch6_match_roster_from_population,
    ch6_opening_classroom_from_population,
    ch6_p_true,
    ch6_param_stats,
    ch6_population_param_cloud_pack,
    ch6_population_heterogeneous_cloud_pack,
    ch6_population_n_sweep_ns,
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

# ch6_89 / ch6_100: five times the dist-reel classroom draws.
CH6_DENSITY_END_D1_N_REEL = int(CH6_POPULATION_DIST_N_REEL) * 5

# Ghost check/cross: rotate hues toward the ghost-line blue.
_CH6_GHOST_HUE_SHIFT = 0.42
_CH6_GHOST_ICON_ALPHA = 0.72
_CH6_GHOST_ICON_ZOOM = 0.18
_GHOST_ICON_CACHE: dict[str, np.ndarray] = {}

# Belief-density colormap on 3D markers and 2D ghost lines (ch6_102+ default).
CH6_BELIEF_DENSITY_KW = dict(
    marker_density_reveal_u=1.0,
    ghost_density_reveal_u=1.0,
)

# Visual state shared by the end of ch6_104 and the start of ch6_112.
CH6_104_112_HANDOFF_ELLIPSOID_KW = dict(
    ellipsoid_lw=1.25,
    ellipsoid_gradient=True,
)


def _ch6_azim_delta_forward_to_match(from_az, to_az):
    """Rotate forward from ``from_az`` until the view matches ``to_az`` (mod 360°)."""
    fa = float(from_az) % 360.0
    ta = float(to_az) % 360.0
    d = (ta - fa) % 360.0
    return 360.0 if d < 1e-9 else d


def _ch6_orbit_view_azim(azim_base, *, move_u=None, return_u=None, orbit_deg=90.0):
    """Ramp azimuth during movement (``move_u``) or unwind at the end (``return_u``)."""
    deg = float(orbit_deg)
    base = float(azim_base)
    if return_u is not None:
        ru = float(np.clip(float(return_u), 0.0, 1.0))
        return base + deg * (1.0 - ru)
    if move_u is not None:
        mu = float(np.clip(float(move_u), 0.0, 1.0))
        return base + deg * mu
    return base


def _ch6_104_112_handoff(stats):
    """Ellipsoid loops + callout for the ch6_104 → ch6_112 seam."""
    mean_pt = np.asarray(stats["mean"], dtype=np.float64)
    loops = ch6_gaussian_ellipsoid_loops(stats["mean"], stats["cov"])
    return dict(
        loops=loops,
        mean_pt=mean_pt,
        gaussian_callout={
            "point": mean_pt,
            "label": "gaussian",
            "color": CH6_VARIANCE_RED,
        },
    )


def install(globals_dict: dict[str, Any]) -> None:
    global _G, _CH6_DUAL_2D_LAYOUT
    _G = globals_dict
    _CH6_DUAL_2D_LAYOUT = None
    _ch6_density_end_d1_pack_cached.cache_clear()
    _ch6_d1_peel_marker_clouds_cached.cache_clear()
    _ch6_dense_n_sweep_states_cached.cache_clear()
    _ch6_wild_n100_reel_pack_cached.cache_clear()
    _ch6_heterogeneous_n100_reel_pack_cached.cache_clear()
    ch6_sampling_cloud.cache_clear()
    _GHOST_ICON_CACHE.clear()
    _G["CH6_EXPORT_SPECS"] = CH6_EXPORT_SPECS
    _G["ch6_export_clip"] = ch6_export_clip


def _g(name: str):
    return _G[name]


def _draft_short(n_full: int, n_draft: int) -> int:
    from ch6_frequentist import _CH3_DRAFT

    return n_draft if _CH3_DRAFT else n_full


def _workshop_short(n_full: int, n_draft: int) -> int:
    """Workshop clips: ~half the legacy HQ frame counts (still smoother than draft)."""
    from ch6_frequentist import _CH3_DRAFT

    if _CH3_DRAFT:
        return n_draft
    return max(n_draft + 2, int(round(float(n_full) * 0.5)))


def _workshop_hold(n: int) -> int:
    """Shorter holds for long workshop reels."""
    from ch6_frequentist import _CH3_DRAFT

    return max(1, int(n) // (4 if _CH3_DRAFT else 2))


CH6_WORKSHOP_MAX_AXIS_CELLS_DRAFT = 14
CH6_WORKSHOP_MAX_AXIS_CELLS_HQ = 18


def _ch6_workshop_max_axis_cells() -> int:
    from ch6_frequentist import _CH3_DRAFT

    return (
        CH6_WORKSHOP_MAX_AXIS_CELLS_DRAFT
        if _CH3_DRAFT
        else CH6_WORKSHOP_MAX_AXIS_CELLS_HQ
    )


def _ch6_cloud_workshop_voxel_cache(study, exam, y, *, bounds):
    """Uncapped cloud-scale grid — same voxel size/frequency as ch6_147 HQ."""
    return _ch6_nll_voxel_cache(study, exam, y, bounds=bounds)


def _ch6_cloud_workshop_voxel_cache_average(classrooms, *, bounds, mu=None, H=None):
    return _ch6_nll_voxel_cache_average(
        classrooms, bounds=bounds, mu=mu, H=H, cache_kw={},
    )


def _ch6_workshop_voxel_cache_kw(*, bounds):
    return dict(
        cubic_cells=True,
        max_axis_cells=_ch6_workshop_max_axis_cells(),
        reference_bounds=bounds,
    )


def _ch6_workshop_voxel_cache(study, exam, y, *, bounds):
    """Capped voxel grid for cloud-scale workshop clips (much faster than full HQ)."""
    return _ch6_nll_voxel_cache(
        study, exam, y, bounds=bounds, **_ch6_workshop_voxel_cache_kw(bounds=bounds),
    )


def _ch6_workshop_voxel_cache_average(classrooms, *, bounds, mu=None, H=None):
    return _ch6_nll_voxel_cache_average(
        classrooms,
        bounds=bounds,
        mu=mu,
        H=H,
        cache_kw=_ch6_workshop_voxel_cache_kw(bounds=bounds),
    )


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


def _finish_wide_2d(img, clip_id: str):
    """Wide 2D + knob row (ch4_02b) — raster only, no tutorial compose crop."""
    del clip_id
    return img


def _fig_to_plot(fig):
    return _g("fig_to_image")(fig, dpi=_g("CH3_ANIM_DPI"))


def _fig_to_plot_wide(fig):
    """Match ch4_02b / ch5 wide-panel animation DPI when available."""
    try:
        dpi = float(_g("_ch4_02b_anim_dpi")())
    except (KeyError, TypeError):
        dpi = float(_g("CH3_ANIM_DPI"))
    return _g("fig_to_image")(fig, dpi=dpi)


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


def _draw_base_dataset(ax, study, exam, y, *, xlim, ylim, alpha=0.95):
    _g("draw_dataset")(ax, study, exam, y, alpha=float(alpha))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _draw_base_dataset_crossfade(
    ax,
    study_a,
    exam_a,
    y_a,
    study_b,
    exam_b,
    y_b,
    *,
    xlim,
    ylim,
    blend_u,
):
    """Crossfade roster A → B on the 2D panel (``blend_u``: 0 = A, 1 = B)."""
    blend_u = float(np.clip(float(blend_u), 0.0, 1.0))
    if blend_u < 1.0 - 1e-6 and len(study_a) > 0:
        _draw_base_dataset(
            ax, study_a, exam_a, y_a,
            xlim=xlim, ylim=ylim, alpha=0.95 * (1.0 - blend_u),
        )
    if blend_u > 1e-6 and len(study_b) > 0:
        _draw_base_dataset(
            ax, study_b, exam_b, y_b,
            xlim=xlim, ylim=ylim, alpha=0.95 * blend_u,
        )


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
    x_lim=None,
    y_lim=None,
    azim=None,
    elev=None,
    camera_zoom=1.0,
    minimal_ui=False,
):
    """Match Ch5 HQ landscape camera (CT view_init + hq elev offset)."""
    dlo1, dhi1, dlo2, dhi2, _, _ = CH5_VIEW_BOUNDS
    if x_lim is not None:
        dlo1, dhi1 = float(x_lim[0]), float(x_lim[1])
    elif xy_lim is not None:
        lo, hi = float(xy_lim[0]), float(xy_lim[1])
        dlo1, dhi1 = lo, hi
    if y_lim is not None:
        dlo2, dhi2 = float(y_lim[0]), float(y_lim[1])
    elif xy_lim is not None and x_lim is None:
        lo, hi = float(xy_lim[0]), float(xy_lim[1])
        dlo2, dhi2 = lo, hi
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
    # Matplotlib "zoom" without changing axis limits: move virtual camera distance.
    zoom = float(max(camera_zoom, 1e-3))
    if hasattr(ax3d, "dist"):
        ax3d.dist = float(np.clip(10.0 / zoom, 3.0, 14.0))


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


def _draw_gaussian_ellipsoids(
    ax3d, loops, *, color=CH6_COV_COLOR, alpha=0.85, lw=1.6, gradient=False,
):
    """Nested 3D ellipsoid wireframes from ``ch6_gaussian_ellipsoid_loops``."""
    if not loops:
        return
    for li, loop in enumerate(loops):
        al = float(alpha) * (0.55 + 0.45 * li / max(len(loops) - 1, 1))
        if gradient:
            # Inner ellipsoids are higher-density mass; keep them darker.
            t = 1.0 - float(li) / max(len(loops) - 1, 1)
            col = ch5_uniform_belief_rgba_at_pdf(t, z_lim=(0.0, 1.0), alpha=al)
        else:
            col = color
        for ring in loop:
            pts = np.asarray(ring, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue
            ax3d.plot(
                pts[0], pts[1], pts[2],
                color=col, lw=lw, alpha=al, zorder=12,
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


def _draw_mu_sigma_labels(ax3d, mean, *, reveal_u=1.0):
    """Simple ``mu`` and ``sigma^2`` labels near the cloud center."""
    u = float(np.clip(reveal_u, 0.0, 1.0))
    if u <= 1e-6:
        return
    mu3 = np.asarray(mean, dtype=np.float64).reshape(3)
    p_mu = mu3 + np.array([0.55, 0.32, 0.18], dtype=np.float64)
    p_sig = mu3 + np.array([0.55, -0.32, -0.18], dtype=np.float64)
    ax3d.plot(
        [mu3[0], p_mu[0]], [mu3[1], p_mu[1]], [mu3[2], p_mu[2]],
        color=CH6_VARIANCE_RED, lw=2.2, alpha=0.95 * u, zorder=14,
    )
    ax3d.plot(
        [mu3[0], p_sig[0]], [mu3[1], p_sig[1]], [mu3[2], p_sig[2]],
        color=CH6_VARIANCE_RED, lw=2.2, alpha=0.95 * u, zorder=14,
    )
    ax3d.scatter(
        [mu3[0]], [mu3[1]], [mu3[2]],
        s=80, c=CH6_VARIANCE_RED, alpha=0.95 * u,
        depthshade=False, edgecolors="white", linewidths=0.6, zorder=15,
    )
    fx, fy = _project_to_axes_fraction(ax3d, mean)
    fx = float(np.clip(fx, 0.08, 0.72))
    fy = float(np.clip(fy, 0.22, 0.88))
    ax3d.text2D(
        fx + 0.03, fy + 0.03, r"$\mu$",
        transform=ax3d.transAxes,
        fontsize=22,
        color=CH6_VARIANCE_RED,
        fontweight="bold",
        alpha=0.95 * u,
        ha="left",
        va="bottom",
        zorder=1000,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=CH6_VARIANCE_RED, alpha=0.92),
    )
    ax3d.text2D(
        fx + 0.03, fy - 0.06, r"$\sigma^2$",
        transform=ax3d.transAxes,
        fontsize=20,
        color=CH6_VARIANCE_RED,
        fontweight="bold",
        alpha=0.90 * u,
        ha="left",
        va="top",
        zorder=1000,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=CH6_VARIANCE_RED, alpha=0.92),
    )


def _draw_mu_label_only(ax3d, mean, *, reveal_u=1.0):
    """``mu`` label only (no ``sigma^2``) near the cloud center."""
    u = float(np.clip(reveal_u, 0.0, 1.0))
    if u <= 1e-6:
        return
    mu3 = np.asarray(mean, dtype=np.float64).reshape(3)
    p_mu = mu3 + np.array([0.55, 0.32, 0.18], dtype=np.float64)
    ax3d.plot(
        [mu3[0], p_mu[0]], [mu3[1], p_mu[1]], [mu3[2], p_mu[2]],
        color=CH6_VARIANCE_RED, lw=2.2, alpha=0.95 * u, zorder=14,
    )
    ax3d.scatter(
        [mu3[0]], [mu3[1]], [mu3[2]],
        s=80, c=CH6_VARIANCE_RED, alpha=0.95 * u,
        depthshade=False, edgecolors="white", linewidths=0.6, zorder=15,
    )
    fx, fy = _project_to_axes_fraction(ax3d, mean)
    fx = float(np.clip(fx, 0.08, 0.72))
    fy = float(np.clip(fy, 0.22, 0.88))
    ax3d.text2D(
        fx + 0.03, fy + 0.03, r"$\mu$",
        transform=ax3d.transAxes,
        fontsize=22,
        color=CH6_VARIANCE_RED,
        fontweight="bold",
        alpha=0.95 * u,
        ha="left",
        va="bottom",
        zorder=1000,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=CH6_VARIANCE_RED, alpha=0.92),
    )


# Marginal Gaussian on coordinate walls for ch6_114.
# (name, pdf_sweep, disp_sweep, height_axis, wall_pin, wall_at_hi, pin_map, height_from_hi)
CH6_114_MARGINAL_SPECS: tuple[tuple[str, int, int, int, int, bool, dict[int, str], bool], ...] = (
    ("w_st", 0, 0, 2, 1, False, {1: "lo", 2: "lo"}, False),
    # w_el: w_st–w_el plane at b=min; sweep w_el, height along w_st
    ("w_el", 1, 1, 0, 2, False, {0: "lo", 2: "lo"}, False),
    # b: along b at w_st=min, w_el=max; curve on w_el–b wall, height w_el
    ("b", 2, 2, 1, 0, False, {0: "lo", 1: "hi"}, True),
)


def _ch6_114_axis_lo_hi(axis_lim):
    return float(axis_lim[0]), float(axis_lim[1])


def _ch6_114_pin_coord(which, axis_lim):
    lo, hi = _ch6_114_axis_lo_hi(axis_lim)
    return hi if str(which) == "hi" else lo


def _ch6_114_apply_proj_pins(pts, pin_map, axis_lim):
    pts = np.asarray(pts, dtype=np.float64).copy()
    for ax, which in pin_map.items():
        pts[:, int(ax)] = _ch6_114_pin_coord(which, axis_lim)
    return pts


def _ch6_114_tight_axis_lim(W, base_lim, *, pad_frac=0.12):
    """Tighter symmetric axis limits around the cloud for axis-zoom."""
    W = np.asarray(W, dtype=np.float64)
    lo_b, hi_b = float(base_lim[0]), float(base_lim[1])
    if len(W) < 2:
        return (lo_b, hi_b)
    lo = float(np.min(W))
    hi = float(np.max(W))
    span = max(hi - lo, 0.35)
    pad = float(pad_frac) * span
    lo_t = max(lo_b, lo - pad)
    hi_t = min(hi_b, hi + pad)
    if hi_t - lo_t < 0.5:
        mid = 0.5 * (lo_t + hi_t)
        lo_t, hi_t = mid - 0.25, mid + 0.25
    return (lo_t, hi_t)


def _ch6_114_marginal_height_scale(W, *, height_axis, axis_lim, height_from_hi=False):
    W = np.asarray(W, dtype=np.float64)
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    vals = W[:, int(height_axis)]
    if height_from_hi:
        span = max(hi - float(np.percentile(vals, 8)), 0.18 * (hi - lo))
    else:
        span = max(float(np.percentile(vals, 92) - lo), 0.18 * (hi - lo))
    return 0.82 * span


def _ch6_114_wall_coords(
    sweep_val,
    height_mag,
    *,
    wall_pin,
    wall_at_hi,
    sweep_axis,
    height_axis,
    axis_lim,
    height_from_hi=False,
):
    lo, hi = _ch6_114_axis_lo_hi(axis_lim)
    p = np.full(3, lo, dtype=np.float64)
    p[int(wall_pin)] = hi if wall_at_hi else lo
    p[int(sweep_axis)] = float(sweep_val)
    if height_from_hi:
        p[int(height_axis)] = hi - float(height_mag)
    else:
        p[int(height_axis)] = lo + float(height_mag)
    return p


def _draw_ch6_114_projected_cloud(
    ax3d,
    W,
    *,
    pin_map,
    axis_lim,
    reveal_u=1.0,
    opacity=1.0,
    marker_s=16,
):
    W = np.asarray(W, dtype=np.float64)
    if len(W) == 0:
        return
    u = float(np.clip(reveal_u, 0.0, 1.0))
    op = float(np.clip(opacity, 0.0, 1.0))
    if u <= 1e-6 or op <= 1e-6:
        return
    pts = _ch6_114_apply_proj_pins(W, pin_map, axis_lim)
    mcols = _ch6_marker_density_colors(W, reveal_u=1.0, z_lim=axis_lim)
    if mcols is not None and u < 1.0:
        mcols = np.asarray(mcols, dtype=np.float64).copy()
        mcols[:, 3] *= u
    if mcols is not None and op < 1.0:
        mcols = np.asarray(mcols, dtype=np.float64).copy()
        mcols[:, 3] *= op
    _draw_markers(
        ax3d, pts, s=float(marker_s), colors=mcols, alpha=u * op,
        z=pts[:, 2], edgecolors="none",
    )


def _draw_ch6_114_marginal_curve(
    ax3d,
    stats,
    *,
    wall_pin,
    wall_at_hi,
    pdf_sweep_axis,
    disp_sweep_axis,
    height_axis,
    axis_lim,
    height_scale,
    reveal_u=1.0,
    opacity=1.0,
    grid_n=120,
    height_from_hi=False,
):
    u = float(np.clip(reveal_u, 0.0, 1.0))
    op = float(np.clip(opacity, 0.0, 1.0))
    if u <= 1e-6 or op <= 1e-6:
        return
    lo, hi = _ch6_114_axis_lo_hi(axis_lim)
    mu = np.asarray(stats["mean"], dtype=np.float64).reshape(3)
    cov = np.asarray(stats["cov"], dtype=np.float64).reshape(3, 3)
    m = float(mu[int(pdf_sweep_axis)])
    v = max(float(cov[int(pdf_sweep_axis), int(pdf_sweep_axis)]), 1e-8)
    xs = np.linspace(lo, hi, int(grid_n))
    pdf = np.exp(-0.5 * (xs - m) ** 2 / v)
    pdf /= max(float(np.max(pdf)), 1e-9)
    n_show = max(2, int(np.ceil(u * len(xs))))
    xs = xs[:n_show]
    pdf = pdf[:n_show]
    pts = np.array([
        _ch6_114_wall_coords(
            x, float(pdf[i]) * float(height_scale),
            wall_pin=wall_pin, wall_at_hi=wall_at_hi,
            sweep_axis=disp_sweep_axis, height_axis=height_axis,
            axis_lim=axis_lim, height_from_hi=height_from_hi,
        )
        for i, x in enumerate(xs)
    ], dtype=np.float64)
    cols = [
        ch5_uniform_belief_rgba_at_pdf(float(t), z_lim=(0.0, 1.0), alpha=0.92)
        for t in pdf
    ]
    for i in range(len(pts) - 1):
        ax3d.plot(
            [pts[i, 0], pts[i + 1, 0]],
            [pts[i, 1], pts[i + 1, 1]],
            [pts[i, 2], pts[i + 1, 2]],
            color=cols[i], lw=2.8, alpha=0.95 * op, zorder=11,
        )


def _draw_ch6_114_mean_line(
    ax3d,
    mu,
    *,
    wall_pin,
    wall_at_hi,
    pdf_sweep_axis,
    disp_sweep_axis,
    mean_line_axis,
    axis_lim,
    reveal_u=1.0,
    opacity=1.0,
    lw=2.6,
    mean_from_hi=False,
):
    u = float(np.clip(reveal_u, 0.0, 1.0))
    op = float(np.clip(opacity, 0.0, 1.0))
    if u <= 1e-6 or op <= 1e-6:
        return
    lo, hi = _ch6_114_axis_lo_hi(axis_lim)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    p0 = np.full(3, lo, dtype=np.float64)
    p0[int(wall_pin)] = hi if wall_at_hi else lo
    p0[int(disp_sweep_axis)] = float(mu[int(pdf_sweep_axis)])
    if mean_from_hi:
        p0[int(mean_line_axis)] = hi
        p1 = p0.copy()
        p1[int(mean_line_axis)] = hi - u * (hi - float(mu[int(mean_line_axis)]))
    else:
        p0[int(mean_line_axis)] = lo
        p1 = p0.copy()
        p1[int(mean_line_axis)] = lo + u * (float(mu[int(mean_line_axis)]) - lo)
    ax3d.plot(
        [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
        color=CH6_VARIANCE_RED, lw=float(lw), alpha=0.95 * op, zorder=13,
    )


def _draw_ch6_114_finale_bridges(ax3d, mu, *, axis_lim, reveal_u=1.0, global_alpha=1.0):
    """Extend the three wall mean lines; orthogonal stubs meet at ``mu``."""
    u = float(np.clip(reveal_u, 0.0, 1.0))
    ga = float(np.clip(global_alpha, 0.0, 1.0))
    if u <= 1e-6 or ga <= 1e-6:
        return
    lo, hi = _ch6_114_axis_lo_hi(axis_lim)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    # w_st wall (w_el = lo): along b at fixed w_st
    a0 = np.array([mu[0], lo, lo], dtype=np.float64)
    a1 = np.array([mu[0], lo, mu[2]], dtype=np.float64)
    # w_el wall (w_st–w_el plane at b=min): along w_st at fixed w_el
    b0 = np.array([lo, mu[1], lo], dtype=np.float64)
    b1 = np.array([mu[0], mu[1], lo], dtype=np.float64)
    # b wall (w_el–b at w_st=min, w_el=max): along w_el at μ_b
    c0 = np.array([lo, hi, mu[2]], dtype=np.float64)
    c1 = np.array([lo, mu[1], mu[2]], dtype=np.float64)
    for p0, p1 in ((a0, a1), (b0, b1), (c0, c1)):
        mid = p0 + u * (p1 - p0)
        _variance_plot_seg(ax3d, p0, mid, color=CH6_VARIANCE_RED, lw=2.8, ls="-", alpha=0.95 * ga)
    _variance_plot_seg(ax3d, a1, mu, color=CH6_VARIANCE_RED, lw=2.0, ls=":", alpha=0.92 * u * ga)
    _variance_plot_seg(ax3d, b1, mu, color=CH6_VARIANCE_RED, lw=2.0, ls=":", alpha=0.92 * u * ga)
    _variance_plot_seg(ax3d, c1, mu, color=CH6_VARIANCE_RED, lw=2.0, ls=":", alpha=0.92 * u * ga)
    ax3d.scatter(
        [mu[0]], [mu[1]], [mu[2]],
        s=70, c=CH6_VARIANCE_RED, alpha=0.95,
        depthshade=False, edgecolors="white", linewidths=0.5, zorder=16,
    )


def _draw_ch6_114_marginal_layers(ax3d, W, stats, layers, *, axis_lim, marker_s=16, global_alpha=1.0):
    ga = float(np.clip(global_alpha, 0.0, 1.0))
    if ga <= 1e-6:
        return
    for layer in layers:
        name, pdf_ax, disp_ax, height_ax, wall_pin, wall_at_hi, pin_map, height_from_hi = layer["spec"]
        op = float(layer.get("alpha", 1.0)) * ga
        if op <= 1e-4:
            continue
        hs = layer.get("height_scale")
        if hs is None:
            hs = _ch6_114_marginal_height_scale(
                W, height_axis=height_ax, axis_lim=axis_lim,
                height_from_hi=height_from_hi,
            )
        mean_line_ax = int(height_ax)
        proj_u = float(layer.get("proj_u", 0.0))
        curve_u = float(layer.get("curve_u", 0.0))
        mean_u = float(layer.get("mean_line_u", 0.0))
        if proj_u > 1e-6:
            _draw_ch6_114_projected_cloud(
                ax3d, W,
                pin_map=pin_map,
                axis_lim=axis_lim,
                reveal_u=proj_u,
                opacity=op,
                marker_s=marker_s,
            )
        if curve_u > 1e-6:
            _draw_ch6_114_marginal_curve(
                ax3d, stats,
                wall_pin=wall_pin, wall_at_hi=wall_at_hi,
                pdf_sweep_axis=pdf_ax, disp_sweep_axis=disp_ax,
                height_axis=height_ax,
                axis_lim=axis_lim, height_scale=hs,
                reveal_u=curve_u,
                opacity=op,
                height_from_hi=height_from_hi,
            )
        if mean_u > 1e-6:
            _draw_ch6_114_mean_line(
                ax3d, stats["mean"],
                wall_pin=wall_pin, wall_at_hi=wall_at_hi,
                pdf_sweep_axis=pdf_ax, disp_sweep_axis=disp_ax,
                mean_line_axis=mean_line_ax,
                axis_lim=axis_lim,
                reveal_u=mean_u,
                opacity=op,
                mean_from_hi=height_from_hi,
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


def _draw_surface(
    ax3d, surf, *, alpha=None, palette="belief", x_lim=None, y_lim=None,
    reveal_origin=None, reveal_u=1.0,
):
    """Likelihood / NLL surface with belief, cloud-density, or NLL heatmap facecolors."""
    W1 = np.asarray(surf["W1"], dtype=np.float64)
    W2 = np.asarray(surf["W2"], dtype=np.float64)
    Z = np.asarray(surf["Z"], dtype=np.float64)
    inside = None
    if x_lim is not None or y_lim is not None:
        inside = np.ones(Z.shape, dtype=bool)
        if x_lim is not None:
            inside &= (W1 >= float(x_lim[0])) & (W1 <= float(x_lim[1]))
        if y_lim is not None:
            inside &= (W2 >= float(y_lim[0])) & (W2 <= float(y_lim[1]))
        Z = np.where(inside, Z, np.nan)
    rev_u = float(np.clip(float(reveal_u), 0.0, 1.0))
    if reveal_origin is not None and rev_u < 1.0 - 1e-6:
        ox = float(np.asarray(reveal_origin, dtype=np.float64).reshape(-1)[0])
        oy = float(np.asarray(reveal_origin, dtype=np.float64).reshape(-1)[1])
        dist = np.hypot(W1 - ox, W2 - oy)
        r_span = float(np.nanmax(dist)) + 1e-9
        radial = dist <= rev_u * r_span
        if inside is None:
            inside = radial
        else:
            inside &= radial
        Z = np.where(inside, Z, np.nan)
    z_hi = max(float(np.nanmax(Z)), float(surf.get("z_lim", (0.0, 1.0))[1]), 1e-6)
    al = float(CH5_BELIEF_SURFACE_ALPHA if alpha is None else alpha)
    if palette == "nll":
        vmin = surf.get("nll_vmin")
        vmax = surf.get("nll_vmax")
        if vmin is None or vmax is None:
            vmin, vmax = _g("ch4_nll_global_scale")()
        fc = np.asarray(
            ch4_nll_heatmap_facecolors(Z, vmin=float(vmin), vmax=float(vmax), alpha=al),
            dtype=np.float64,
        )
    elif palette == "cloud":
        fc = _cloud_density_facecolors(Z, z_lim=(0.0, z_hi), surface_alpha=al)
    else:
        fc = ch5_uniform_belief_facecolors(Z, z_lim=(0.0, z_hi), surface_alpha=al)
    if inside is not None:
        fc = np.asarray(fc, dtype=np.float64).copy()
        fc[~inside] = 0.0
    ch5_plot_belief_surface_with_grid(
        ax3d, W1, W2, Z, facecolors=fc, zorder=1.0, antialiased=True,
    )


def _ch6_nll_surface_w12(study, exam, y, *, b_fixed=0.0, bounds=None, grid=48):
    """Classroom NLL surface on (w_ST, w_EL) with b fixed."""
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if bounds is None:
        bounds = CH6_VIEW_BOUNDS_W12
    Xd = ch6_design(study, exam)
    w_hat, _ = ch6_fit_dataset(study, exam, y, ridge=CH6_RIDGE)
    w_hat = np.asarray(w_hat, dtype=np.float64).reshape(3)
    w1, w2, z_lr = ch6_lr_grid_w12(
        Xd, y, w_hat, ridge=CH6_RIDGE, b_fixed=float(b_fixed), bounds=bounds, grid=int(grid),
    )
    W1, W2 = np.meshgrid(w1, w2)
    nll_hat = float(ch6_nll(w_hat, Xd, y, ridge=CH6_RIDGE))
    Z = nll_hat + 0.5 * np.asarray(z_lr, dtype=np.float64)
    vmin, vmax = _g("ch4_nll_global_scale")()
    z_lo = float(np.nanmin(Z))
    z_hi = float(np.nanmax(Z))
    return {
        "w1": w1,
        "w2": w2,
        "W1": W1,
        "W2": W2,
        "Z": Z,
        "Z_lr": z_lr,
        "nll_vmin": float(vmin),
        "nll_vmax": float(vmax),
        "z_lim": (z_lo, z_hi),
    }


def _ch6_nll_surface_w12_argmin(surf):
    """(w_ST, w_EL, b=0) at the minimum of a classroom NLL bowl slice."""
    Z = np.asarray(surf["Z"], dtype=np.float64)
    W1 = np.asarray(surf["W1"], dtype=np.float64)
    W2 = np.asarray(surf["W2"], dtype=np.float64)
    k = np.unravel_index(int(np.nanargmin(Z)), Z.shape)
    return np.array([float(W1[k]), float(W2[k]), 0.0], dtype=np.float64)


def _ch6_quadratic_w12_surface(center_xy, H, bounds, grid, *, z_base=0.0, palette="belief"):
    """Quadratic bowl Z = z_base + ½ δᵀ H δ — for steep-vs-flat curvature demos."""
    w1 = np.linspace(float(bounds[0]), float(bounds[1]), int(grid))
    w2 = np.linspace(float(bounds[2]), float(bounds[3]), int(grid))
    W1, W2 = np.meshgrid(w1, w2)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    d1 = W1 - cx
    d2 = W2 - cy
    Hm = np.asarray(H, dtype=np.float64).reshape(2, 2)
    Z = float(z_base) + 0.5 * (
        Hm[0, 0] * d1 * d1 + 2.0 * Hm[0, 1] * d1 * d2 + Hm[1, 1] * d2 * d2
    )
    if str(palette) == "belief":
        Z = float(np.nanmax(Z)) - Z
    z_lo = float(np.nanmin(Z))
    z_hi = float(np.nanmax(Z))
    vmin, vmax = _g("ch4_nll_global_scale")()
    return {
        "w1": w1,
        "w2": w2,
        "W1": W1,
        "W2": W2,
        "Z": Z,
        "z_lim": (z_lo, z_hi),
        "nll_vmin": float(vmin),
        "nll_vmax": float(vmax),
        "palette": str(palette),
    }


def _ch6_surface_rotate_w12(surf, bounds, grid, *, angle_deg=90.0):
    """Rotate a w12 surface counter-clockwise in weight space and resample on a grid."""
    w1n = np.linspace(float(bounds[0]), float(bounds[1]), int(grid))
    w2n = np.linspace(float(bounds[2]), float(bounds[3]), int(grid))
    W1n, W2n = np.meshgrid(w1n, w2n)
    cx = 0.5 * (float(bounds[0]) + float(bounds[1]))
    cy = 0.5 * (float(bounds[2]) + float(bounds[3]))
    th = np.deg2rad(float(angle_deg))
    c, s = np.cos(th), np.sin(th)
    dx = W1n - cx
    dy = W2n - cy
    xo = c * dx + s * dy + cx
    yo = -s * dx + c * dy + cy
    Zn = np.empty_like(W1n)
    for i in range(W1n.shape[0]):
        for j in range(W1n.shape[1]):
            Zn[i, j] = _surface_z_at(surf, float(xo[i, j]), float(yo[i, j]))
    z_lo, z_hi = float(np.nanmin(Zn)), float(np.nanmax(Zn))
    vmin = surf.get("nll_vmin")
    vmax = surf.get("nll_vmax")
    if vmin is None or vmax is None:
        vmin, vmax = _g("ch4_nll_global_scale")()
    return {
        "w1": w1n,
        "w2": w2n,
        "W1": W1n,
        "W2": W2n,
        "Z": Zn,
        "z_lim": (z_lo, z_hi),
        "nll_vmin": float(vmin),
        "nll_vmax": float(vmax),
    }


def _ch6_surf_grad_w12(surf, w1, w2, *, h=None):
    h = float(_avg_surf_fd_step(surf) if h is None else h)
    fx = (
        _surface_z_at(surf, float(w1) + h, float(w2))
        - _surface_z_at(surf, float(w1) - h, float(w2))
    ) / (2.0 * h)
    fy = (
        _surface_z_at(surf, float(w1), float(w2) + h)
        - _surface_z_at(surf, float(w1), float(w2) - h)
    ) / (2.0 * h)
    return np.array([fx, fy], dtype=np.float64)


def _ch6_newton_dir_w12_surface(surf, w, *, ridge_floor=0.12):
    """Newton direction from a gridded NLL surface at ``w``."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    g = _ch6_surf_grad_w12(surf, w[0], w[1])
    H = _ch6_nll_surf_hess_w12(surf, float(w[0]), float(w[1]))
    H = H + float(ridge_floor) * np.eye(2, dtype=np.float64)
    try:
        d = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        gn = float(np.linalg.norm(g))
        d = -g if gn > 1e-12 else np.zeros(2, dtype=np.float64)
    return d.astype(np.float64)


def _ch6_clip_w12_xy(pt, axis_lim):
    """Clip (w_ST, w_EL) to the population axis range."""
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    p = np.asarray(pt, dtype=np.float64).reshape(3).copy()
    p[0] = float(np.clip(p[0], lo, hi))
    p[1] = float(np.clip(p[1], lo, hi))
    p[2] = 0.0
    return p


def _ch6_smoothstep01(u):
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _ch6_resample_surface_w12(surf, bounds, grid):
    """Resample a w12 surface onto a new grid (no rotation)."""
    w1 = np.linspace(float(bounds[0]), float(bounds[1]), int(grid))
    w2 = np.linspace(float(bounds[2]), float(bounds[3]), int(grid))
    W1, W2 = np.meshgrid(w1, w2)
    Zn = np.empty_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            Zn[i, j] = _surface_z_at(surf, float(W1[i, j]), float(W2[i, j]))
    z_lo, z_hi = float(np.nanmin(Zn)), float(np.nanmax(Zn))
    vmin = surf.get("nll_vmin")
    vmax = surf.get("nll_vmax")
    if vmin is None or vmax is None:
        vmin, vmax = _g("ch4_nll_global_scale")()
    return {
        "w1": w1,
        "w2": w2,
        "W1": W1,
        "W2": W2,
        "Z": Zn,
        "z_lim": (z_lo, z_hi),
        "nll_vmin": float(vmin),
        "nll_vmax": float(vmax),
    }


def _ch6_geom_to_belief_vis(geom):
    """Invert NLL heights for the belief colormap display."""
    Z = np.asarray(geom["Z"], dtype=np.float64)
    vis = dict(geom)
    Z_vis = np.nanmax(Z) - Z
    vis["Z"] = Z_vis
    vis["z_lim"] = (float(np.nanmin(Z_vis)), float(np.nanmax(Z_vis)))
    vis["palette"] = "belief"
    return vis


def _ch6_w12_morph_geom(geom_a, geom_b, u):
    """Blend two w12 surfaces on a shared grid (smooth bend)."""
    u = float(np.clip(float(u), 0.0, 1.0))
    Za = np.asarray(geom_a["Z"], dtype=np.float64)
    Zb = np.asarray(geom_b["Z"], dtype=np.float64)
    out = dict(geom_a)
    Z = (1.0 - u) * Za + u * Zb
    out["Z"] = Z
    out["z_lim"] = (float(np.nanmin(Z)), float(np.nanmax(Z)))
    return out


def _ch6_w12_pick_rotated_geom(class_surf, mu, w_class, bounds, grid):
    """Pick ±90° so classroom NLL exceeds average (black below red visually)."""
    mu2 = np.asarray(mu[:2], dtype=np.float64)
    wc2 = np.asarray(w_class[:2], dtype=np.float64)
    best_geom = None
    best_score = -np.inf
    for ang in (90.0, -90.0):
        geom = _ch6_surface_rotate_w12(
            class_surf, bounds, grid, angle_deg=float(ang),
        )
        z_mu = float(_surface_z_at(geom, float(mu2[0]), float(mu2[1])))
        z_cl = float(_surface_z_at(geom, float(wc2[0]), float(wc2[1])))
        score = z_cl - z_mu
        if score > best_score:
            best_score = score
            best_geom = geom
    return best_geom


def _ch6_w12_morph_scene_pack(class_surf, mu_w12, w_class, g_newton, bounds, grid):
    """Classroom NLL → 90°-rotated bend; markers stay at mu / classroom xy."""
    class_zoom = _ch6_resample_surface_w12(class_surf, bounds, grid)
    rot_geom = _ch6_w12_pick_rotated_geom(
        class_surf, mu_w12, w_class, bounds, grid,
    )
    H_w12 = _ch6_nll_surf_hess_w12(
        rot_geom, float(mu_w12[0]), float(mu_w12[1]),
    )
    return {
        "class_zoom_geom": class_zoom,
        "class_zoom_vis": _ch6_geom_to_belief_vis(class_zoom),
        "steep_geom": rot_geom,
        "steep_surf": _ch6_geom_to_belief_vis(rot_geom),
        "steep_H_w12": H_w12,
        "g_newton": np.asarray(g_newton, dtype=np.float64).reshape(2).copy(),
    }


def _ch6_w12_plunge_surface(
    class_surf, p0, p1, bounds, grid, *,
    z_bump=0.045, plunge_curv_mult=4.5,
):
    """Taylor match at ``p0`` (classroom NLL), then plunge to a valley at ``p1``."""
    p0 = np.asarray(p0[:2], dtype=np.float64)
    p1 = np.asarray(p1[:2], dtype=np.float64)
    z0 = float(_surface_z_at(class_surf, float(p0[0]), float(p0[1]))) + float(z_bump)
    g0 = _ch6_surf_grad_w12(class_surf, float(p0[0]), float(p0[1]))
    H0 = _ch6_nll_surf_hess_w12(class_surf, float(p0[0]), float(p0[1]))

    delta = p1 - p0
    L = max(float(np.linalg.norm(delta)), 0.12)
    u_dir = delta / L
    v_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float64)

    d1 = p1 - p0
    z_taylor_p1 = z0 + float(g0 @ d1) + 0.5 * float(d1 @ H0 @ d1)
    z_class_ref = float(_surface_z_at(class_surf, float(p1[0]), float(p1[1])))
    z_valley = min(z_taylor_p1 - 0.30, z_class_ref - 0.08)
    drop_amp = max(z_taylor_p1 - z_valley, 0.06)

    evals = np.linalg.eigvalsh(H0)
    k_steep = float(plunge_curv_mult) * max(float(np.max(evals)), 1.2)

    w1 = np.linspace(float(bounds[0]), float(bounds[1]), int(grid))
    w2 = np.linspace(float(bounds[2]), float(bounds[3]), int(grid))
    W1, W2 = np.meshgrid(w1, w2)
    dx = W1 - p0[0]
    dy = W2 - p0[1]
    s = dx * u_dir[0] + dy * u_dir[1]
    t = dx * v_dir[0] + dy * v_dir[1]
    h00, h01, h11 = float(H0[0, 0]), float(H0[0, 1]), float(H0[1, 1])
    z_taylor = (
        z0 + g0[0] * dx + g0[1] * dy
        + 0.5 * (h00 * dx * dx + 2.0 * h01 * dx * dy + h11 * dy * dy)
    )
    phi = _ch6_smoothstep01(np.clip(s / L, 0.0, 1.15))
    Z = z_taylor - drop_amp * phi + 0.5 * k_steep * t * t * phi
    z_lo = float(np.nanmin(Z))
    z_hi = float(np.nanmax(Z))
    vmin, vmax = _g("ch4_nll_global_scale")()
    return {
        "w1": w1,
        "w2": w2,
        "W1": W1,
        "W2": W2,
        "Z": Z,
        "z_lim": (z_lo, z_hi),
        "nll_vmin": float(vmin),
        "nll_vmax": float(vmax),
    }


def _ch6_w12_plunge_surface_pack(class_surf, p0, p1, g_newton, bounds, grid):
    """Plunge landscape: same local bowl at red, valley at classroom."""
    geom = _ch6_w12_plunge_surface(
        class_surf, p0, p1, bounds, grid,
        plunge_curv_mult=CH6_W12_PUSH_HIGH_CURV_MULT,
    )
    Z = np.asarray(geom["Z"], dtype=np.float64)
    vis = dict(geom)
    Z_vis = np.nanmax(Z) - Z
    vis["Z"] = Z_vis
    vis["z_lim"] = (float(np.nanmin(Z_vis)), float(np.nanmax(Z_vis)))
    vis["palette"] = "belief"
    H_w12 = _ch6_nll_surf_hess_w12(geom, float(p0[0]), float(p0[1]))
    return {
        "steep_geom": geom,
        "steep_surf": vis,
        "p0": np.asarray(p0, dtype=np.float64).reshape(3),
        "p1": np.asarray(p1, dtype=np.float64).reshape(3),
        "H_w12": H_w12,
        "g_newton": np.asarray(g_newton, dtype=np.float64).reshape(2).copy(),
    }


def _ch6_w12_reposition_w12_pair(mu, w_class, axis_lim, zoom_bounds):
    """Shift the pair perpendicular to their chord, staying inside zoom + axis."""
    mu2 = np.asarray(mu[:2], dtype=np.float64)
    wc2 = np.asarray(w_class[:2], dtype=np.float64)
    delta = wc2 - mu2
    L = float(np.linalg.norm(delta))
    if L < 1e-9:
        u_dir = np.array([1.0, 0.0], dtype=np.float64)
        v_dir = np.array([0.0, 1.0], dtype=np.float64)
        L = 0.2
    else:
        u_dir = delta / L
        v_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float64)
    p0 = mu2 + 0.34 * L * v_dir
    p1 = wc2 - 0.24 * L * v_dir
    lo_ax, hi_ax = float(axis_lim[0]), float(axis_lim[1])
    zlo1, zhi1, zlo2, zhi2 = (float(v) for v in zoom_bounds)

    def _clip2(p):
        return np.array([
            float(np.clip(p[0], max(lo_ax, zlo1), min(hi_ax, zhi1))),
            float(np.clip(p[1], max(lo_ax, zlo2), min(hi_ax, zhi2))),
        ], dtype=np.float64)

    p0 = _clip2(p0)
    p1 = _clip2(p1)
    return (
        np.array([p0[0], p0[1], 0.0], dtype=np.float64),
        np.array([p1[0], p1[1], 0.0], dtype=np.float64),
    )


def _ch6_newton_walk_w12_surface(surf, p0, p1, *, n_frames, step_eta=None):
    """Discrete Newton descent on a gridded w12 surface toward ``p1``."""
    p = np.asarray(p0[:2], dtype=np.float64).reshape(2).copy()
    target = np.asarray(p1[:2], dtype=np.float64).reshape(2)
    eta = float(CH6_W12_PUSH_STEP_ETA * 1.2 if step_eta is None else step_eta)
    lo1, hi1, lo2, hi2 = CH6_VIEW_BOUNDS_W12
    path = [p.copy()]
    n_step = max(int(n_frames) - 1, 1)
    for _ in range(n_step):
        w3 = np.array([p[0], p[1], 0.0], dtype=np.float64)
        step = _ch6_newton_dir_w12_surface(surf, w3)
        sn = float(np.linalg.norm(step))
        if sn < 1e-12:
            break
        move = step / sn * min(sn, 0.11)
        p = p + eta * move
        p[0] = float(np.clip(p[0], lo1, hi1))
        p[1] = float(np.clip(p[1], lo2, hi2))
        path.append(p.copy())
        if float(np.linalg.norm(p - target)) < 0.028:
            break
    if float(np.linalg.norm(path[-1] - target)) > 1e-4:
        path.append(target.copy())
    return np.asarray(path, dtype=np.float64)


def _ch6_curvature_push_path(p0, g_push, H, *, n_frames, step_eta=None, curv_mult=1.0):
    """Fixed-direction push with step size decaying ∝ exp(−½ δᵀ H δ) from the start."""
    p0 = np.asarray(p0, dtype=np.float64).reshape(2)
    g = np.asarray(g_push, dtype=np.float64).reshape(2)
    gn = float(np.linalg.norm(g))
    if gn < 1e-12:
        return np.tile(p0.reshape(1, 2), (max(int(n_frames), 2), 1))
    u_dir = g / gn
    H_eff = float(curv_mult) * np.asarray(H, dtype=np.float64).reshape(2, 2)
    evals = np.linalg.eigvalsh(H_eff)
    floor = max(0.08, -float(np.min(evals)) + 0.08)
    H_eff = H_eff + floor * np.eye(2, dtype=np.float64)
    lo1, hi1, lo2, hi2 = CH6_VIEW_BOUNDS_W12
    eta = float(CH6_W12_PUSH_STEP_ETA if step_eta is None else step_eta)
    path = [p0.copy()]
    p = p0.copy()
    n_step = max(int(n_frames) - 1, 1)
    for _ in range(n_step):
        d = p - p0
        decay = float(np.exp(-0.62 * float(d @ H_eff @ d)))
        p = p + eta * decay * u_dir
        p[0] = float(np.clip(p[0], lo1, hi1))
        p[1] = float(np.clip(p[1], lo2, hi2))
        path.append(p.copy())
    return np.asarray(path, dtype=np.float64)


def _ch6_w12_push_scene_pack():
    """Shared classroom pose, NLL bowl, and zoom window for ch6_131 / ch6_132."""
    from ch6_frequentist import _CH3_DRAFT

    pose = _ch6_matrix_contrast_cloud()
    info = pose["info"]
    mu = np.asarray(pose["mu"], dtype=np.float64).reshape(3)
    W = np.asarray(pose["W"], dtype=np.float64)
    study = np.asarray(info["cs"], dtype=np.float64)
    exam = np.asarray(info["ce"], dtype=np.float64)
    y = np.asarray(info["cy"], dtype=np.float64).reshape(-1)
    w_knob = np.asarray(info["w_knob"], dtype=np.float64).reshape(3)
    mu_w12 = mu.copy()
    mu_w12[2] = 0.0
    axis_lim = pose["axis_lim"]
    ax_lo, ax_hi = float(axis_lim[0]), float(axis_lim[1])
    w12_full = (ax_lo, ax_hi, ax_lo, ax_hi)
    grid = 28 if _CH3_DRAFT else 64
    class_surf = _ch6_nll_surface_w12(
        study, exam, y, b_fixed=0.0, bounds=w12_full, grid=grid,
    )
    w_class = _ch6_nll_surface_w12_argmin(class_surf)
    zoom_bounds = _ch6_voxel_bounds_w12_between_points(mu_w12, w_class, axis_lim, pad_frac=0.15)
    w12_zoom = (
        float(zoom_bounds[0]), float(zoom_bounds[1]),
        float(zoom_bounds[2]), float(zoom_bounds[3]),
    )
    Xd = ch6_design(study, exam)
    H_w12 = _ch6_nll_surf_hess_w12(class_surf, float(mu_w12[0]), float(mu_w12[1]))
    g_newton = _ch6_newton_dir_w12(mu_w12, Xd, y)
    steep_pack = _ch6_w12_morph_scene_pack(
        class_surf, mu_w12, w_class, g_newton, w12_zoom, grid,
    )
    return {
        "pose": pose,
        "info": info,
        "mu": mu,
        "W": W,
        "study": study,
        "exam": exam,
        "y": y,
        "w_knob": w_knob,
        "mu_w12": mu_w12,
        "w_class": w_class,
        "axis_lim": axis_lim,
        "w12_full": w12_full,
        "w12_zoom": w12_zoom,
        "class_surf": class_surf,
        "class_zoom_geom": steep_pack["class_zoom_geom"],
        "class_zoom_vis": steep_pack["class_zoom_vis"],
        "steep_surf": steep_pack["steep_surf"],
        "steep_geom": steep_pack["steep_geom"],
        "steep_H_w12": steep_pack["steep_H_w12"],
        "Xd": Xd,
        "H_w12": H_w12,
        "g_newton": g_newton,
        "grid": grid,
    }


def _ch6_nll_surf_hess_w12(surf, w1, w2, *, h=None):
    """Hessian of NLL on a classroom NLL surface grid."""
    h = float(_avg_surf_fd_step(surf) if h is None else h)

    def nll(x, y):
        return float(_surface_z_at(surf, x, y))

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
    evals = np.linalg.eigvalsh(H)
    floor = max(0.35, -float(np.min(evals)) + 0.08)
    return H + floor * np.eye(2, dtype=np.float64)


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


def _draw_markers(
    ax3d,
    weights,
    *,
    color=CH6_CLOUD_COLOR,
    colors=None,
    s=28,
    alpha=0.85,
    z=None,
    edgecolors=None,
):
    Ws = np.asarray(weights, dtype=np.float64)
    if Ws.size == 0:
        return
    if z is None:
        zz = np.full(len(Ws), 0.02)
    else:
        zz = np.asarray(z, dtype=np.float64)
    c = np.asarray(colors, dtype=np.float64) if colors is not None else color
    scatter_kw = dict(
        s=s, c=c, depthshade=False, edgecolors="white", linewidths=0.4, zorder=8,
    )
    if edgecolors is not None:
        scatter_kw["edgecolors"] = edgecolors
        if edgecolors == "none":
            scatter_kw["linewidths"] = 0.0
    elif isinstance(c, np.ndarray) and c.ndim == 2 and c.shape[1] == 4:
        # Per-marker alpha lives in the RGBA array; a scalar alpha overrides it.
        if float(np.max(c[:, 3])) <= 1e-6:
            return
        if float(np.mean(c[:, 3])) < 0.35:
            scatter_kw["edgecolors"] = "none"
            scatter_kw["linewidths"] = 0.0
    else:
        scatter_kw["alpha"] = alpha
    ax3d.scatter(Ws[:, 0], Ws[:, 1], zz, **scatter_kw)


def _ch6_kde_density_values(W, *, query=None):
    """Local sampling density at each row of ``query`` (defaults to ``W``)."""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or len(W) < 5:
        n = len(W) if query is None else len(np.asarray(query))
        return np.ones(max(n, 1), dtype=np.float64)
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(W.T)
    Q = W if query is None else np.asarray(query, dtype=np.float64).reshape(-1, 3)
    return np.maximum(kde(Q.T), 1e-12)


def _ch6_density_belief_rgba(density_vals, *, z_lim=None, alpha=0.92):
    """Map local densities onto the Ch5 belief heatmap (pink → blue)."""
    if z_lim is None:
        z_lim = ch5_uniform_belief_z_lim()
    z_lo, z_hi = float(z_lim[0]), float(z_lim[1])
    d = np.asarray(density_vals, dtype=np.float64).reshape(-1)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-12:
        pdf = np.full(len(d), 0.5 * (z_lo + z_hi), dtype=np.float64)
    else:
        t = (d - lo) / (hi - lo)
        pdf = z_lo + t * (z_hi - z_lo)
    return np.asarray([
        ch5_uniform_belief_rgba_at_pdf(float(p), z_lim=z_lim, alpha=alpha)
        for p in pdf
    ], dtype=np.float64)


def _ch6_reveal_belief_colors(full_rgba, reveal_u, *, base_rgb=None, base_alpha=0.85):
    """Crossfade from the default cloud grey into belief-density colors."""
    u = float(np.clip(float(reveal_u), 0.0, 1.0))
    if u <= 1e-6:
        return None
    if base_rgb is None:
        base_rgb = np.array(plt.matplotlib.colors.to_rgb(CH6_CLOUD_COLOR), dtype=np.float64)
    full_rgba = np.asarray(full_rgba, dtype=np.float64)
    if u >= 1.0 - 1e-6:
        return full_rgba
    out = np.empty_like(full_rgba)
    for i, rgba in enumerate(full_rgba):
        rgb = base_rgb * (1.0 - u) + rgba[:3] * u
        a = base_alpha * (1.0 - u) + float(rgba[3]) * u
        out[i] = (*rgb, a)
    return out


def _ch6_marker_density_colors(W, *, reveal_u=1.0, z_lim=None, alpha=0.92):
    """Per-marker RGBA from KDE density × Ch5 belief colormap."""
    W = np.asarray(W, dtype=np.float64)
    if len(W) == 0:
        return None
    full = _ch6_density_belief_rgba(_ch6_kde_density_values(W), z_lim=z_lim, alpha=alpha)
    return _ch6_reveal_belief_colors(full, reveal_u, base_alpha=0.85)


def _ch6_ghost_density_colors(ghost_ws, markers, *, reveal_u=1.0, z_lim=None, alpha=0.72):
    """Per-ghost-line hex colors from KDE density at each ghost weight."""
    if not ghost_ws or markers is None:
        return None
    u = float(np.clip(float(reveal_u), 0.0, 1.0))
    if u <= 1e-6:
        return None
    W = np.asarray(markers, dtype=np.float64)
    ghosts = [np.asarray(wg, dtype=np.float64).reshape(3) for wg in ghost_ws]
    if len(W) < 5:
        return None
    dens = _ch6_kde_density_values(W, query=np.vstack(ghosts))
    full = _ch6_density_belief_rgba(dens, z_lim=z_lim, alpha=alpha)
    ghost_base = np.array(plt.matplotlib.colors.to_rgb(CH6_GHOST_COLOR), dtype=np.float64)
    cols: list[str] = []
    for rgba in full:
        if u >= 1.0 - 1e-6:
            rgb = rgba[:3]
            a = float(rgba[3])
        else:
            rgb = ghost_base * (1.0 - u) + rgba[:3] * u
            a = 0.55 * (1.0 - u) + float(rgba[3]) * u
        cols.append(plt.matplotlib.colors.to_hex((*rgb, a), keep_alpha=True))
    return cols


def _cloud_density_rgba_at_t(t, *, alpha_scale=0.92):
    """Histogram-bin color from the same belief colormap as cloud markers."""
    t = float(np.clip(t, 0.0, 1.0))
    return ch5_uniform_belief_rgba_at_pdf(
        t,
        z_lim=(0.0, 1.0),
        alpha=(0.22 + 0.72 * t) * float(alpha_scale),
    )


def _draw_gaussian_callout(
    fig, ax3d, point, *, label="gaussian", color="#ef4444", label_fig=None, alpha=1.0,
    label_ha="right", label_va="top",
):
    """White callout label + arrow, like chapter-5 'we are here' placement."""
    from matplotlib.patches import FancyArrowPatch

    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 1e-4:
        return
    p = np.asarray(point, dtype=np.float64).reshape(3)
    fig.canvas.draw()
    px, py = _g("ch3_ax3d_to_fig_xy")(fig, ax3d, float(p[0]), float(p[1]), float(p[2]))
    if label_fig is None:
        label_fig = _g("CH3_LIK_RIDGE_LABEL_FIG")
    lx, ly = float(label_fig[0]), float(label_fig[1])
    ha = str(label_ha)
    va = str(label_va)
    fig.text(
        lx, ly, str(label),
        transform=fig.transFigure,
        fontsize=_g("CH3_LIK_RIDGE_HERE_FS"),
        color="#111111",
        fontfamily="DejaVu Sans",
        fontweight="bold",
        ha=ha,
        va=va,
        zorder=60,
        alpha=a,
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", edgecolor=str(color),
            alpha=0.97 * a,
        ),
    )
    dx, dy = float(px) - lx, float(py) - ly
    d = float(np.hypot(dx, dy)) + 1e-9
    sx = lx + 0.06 * dx / d
    sy = ly + 0.06 * dy / d
    arrow = FancyArrowPatch(
        (sx, sy), (px, py),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.4,
        color=str(color),
        shrinkA=8,
        shrinkB=8,
        zorder=59,
        alpha=a,
    )
    fig.patches.append(arrow)


def _ch6_callout_safe_bounds(*, for_tutorial=True):
    """Figure-fraction box for labels — survives tutorial compose without rail overlap."""
    if for_tutorial:
        return (0.05, 0.58, 0.82, 0.93)
    return (0.04, 0.55, 0.90, 0.95)


# ch6_160 sandwich estimator — fixed label slots (figure fraction, left of tutorial rail).
CH6_SANDWICH_160_PER_STUDENT_LABEL_FIG = (0.06, 0.938)
CH6_SANDWICH_160_HESS_LABEL_FIG = (0.38, 0.915)
CH6_SANDWICH_160_RIM_LABEL_DROP = 0.022
CH6_SANDWICH_160_TUTORIAL_RAIL_X = 0.80


def _ch6_callout_anchor_placement(fig, ax3d, anchor):
    """Fixed rim slots for ch6_160 (stay on-screen, clear of 2D panel + tutorial rails)."""
    fig.canvas.draw()
    pos = ax3d.get_position()
    x0, y0, w, h = float(pos.x0), float(pos.y0), float(pos.width), float(pos.height)
    ins = 0.028
    rail = float(CH6_SANDWICH_160_TUTORIAL_RAIL_X)
    a = str(anchor)
    drop = float(CH6_SANDWICH_160_RIM_LABEL_DROP)
    if a == "sandwich_avg_tr":
        lx = min(x0 + w - ins, rail - 0.02)
        ly = y0 + h - ins - drop
        return {"label_fig": (lx, ly), "label_ha": "right", "label_va": "top"}
    if a == "sandwich_class_tl":
        lx = max(x0 + ins, float(pos.x0) + ins)
        ly = y0 + h - ins - drop
        return {"label_fig": (lx, ly), "label_ha": "left", "label_va": "top"}
    if a == "sandwich_hess":
        lx, ly = CH6_SANDWICH_160_HESS_LABEL_FIG
        return {"label_fig": (float(lx), float(ly)), "label_ha": "left", "label_va": "top"}
    if a == "sandwich_grad":
        lx = min(x0 + 0.55 * w, rail - 0.22)
        ly = y0 + h - ins
        return {"label_fig": (lx, ly), "label_ha": "center", "label_va": "top"}
    return _ch6_single_callout_placement(fig, ax3d, (0.0, 0.0, 0.0))


def _ch6_diagonal_callout_placement(px, py, bounds):
    """Place label at the safe-corner farthest from the target (diagonal callout)."""
    x0, y0, x1, y1 = (float(v) for v in bounds)
    corners = (
        (x0, y1, "left", "top"),
        (x1, y1, "right", "top"),
        (x0, y0, "left", "bottom"),
        (x1, y0, "right", "bottom"),
    )
    best = max(corners, key=lambda c: (float(px) - c[0]) ** 2 + (float(py) - c[1]) ** 2)
    return (best[0], best[1]), best[2], best[3]


def _ch6_pair_diagonal_callout_placements(fig, ax3d, pa, pb, *, bounds=None):
    """Diagonal opposite corners for a pair of 3D callouts."""
    pa = np.asarray(pa, dtype=np.float64).reshape(3)
    pb = np.asarray(pb, dtype=np.float64).reshape(3)
    bounds = bounds if bounds is not None else _ch6_callout_safe_bounds()
    fig.canvas.draw()
    pxa, pya = _g("ch3_ax3d_to_fig_xy")(fig, ax3d, float(pa[0]), float(pa[1]), float(pa[2]))
    pxb, pyb = _g("ch3_ax3d_to_fig_xy")(fig, ax3d, float(pb[0]), float(pb[1]), float(pb[2]))
    x0, y0, x1, y1 = (float(v) for v in bounds)
    corners = (
        (x0, y1, "left", "top"),
        (x1, y1, "right", "top"),
        (x0, y0, "left", "bottom"),
        (x1, y0, "right", "bottom"),
    )
    scored_a = sorted(
        [
            ((float(pxa) - c[0]) ** 2 + (float(pya) - c[1]) ** 2, c)
            for c in corners
        ],
        reverse=True,
    )
    scored_b = sorted(
        [
            ((float(pxb) - c[0]) ** 2 + (float(pyb) - c[1]) ** 2, c)
            for c in corners
        ],
        reverse=True,
    )
    ca = scored_a[0][1]
    cb = scored_b[0][1]
    if ca[:2] == cb[:2] and len(scored_b) > 1:
        cb = scored_b[1][1]
    return (
        {"label_fig": (ca[0], ca[1]), "label_ha": ca[2], "label_va": ca[3]},
        {"label_fig": (cb[0], cb[1]), "label_ha": cb[2], "label_va": cb[3]},
    )


def _ch6_data_panel_callout_placement(fig, ax_data, xy):
    """Diagonal corner inside the data panel, farthest from the arrow tip."""
    x, y = float(xy[0]), float(xy[1])
    fig.canvas.draw()
    px, py = ax_data.transData.transform((x, y))
    px, py = fig.transFigure.inverted().transform((px, py))
    pos = ax_data.get_position()
    x0 = float(pos.x0) + 0.04 * float(pos.width)
    x1 = float(pos.x0) + 0.96 * float(pos.width)
    y0 = float(pos.y0) + 0.10 * float(pos.height)
    y1 = float(pos.y1) - 0.04 * float(pos.height)
    fig_xy, ha, va = _ch6_diagonal_callout_placement(px, py, (x0, y0, x1, y1))
    return fig_xy, ha, va


def _draw_data_panel_callout(
    fig, ax_data, xy, *, label="", color="#ef4444", label_fig=None, alpha=1.0,
    label_ha="left", label_va="center",
):
    """White callout + arrow on the 2D data panel (ch6_160 per-student Ĵ label)."""
    from matplotlib.patches import FancyArrowPatch

    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 1e-4:
        return
    x, y = float(xy[0]), float(xy[1])
    fig.canvas.draw()
    px, py = ax_data.transData.transform((x, y))
    px, py = fig.transFigure.inverted().transform((px, py))
    if label_fig is None:
        label_fig, label_ha, label_va = _ch6_data_panel_callout_placement(fig, ax_data, xy)
    lx, ly = float(label_fig[0]), float(label_fig[1])
    ha = str(label_ha)
    va = str(label_va)
    fig.text(
        lx, ly, str(label),
        transform=fig.transFigure,
        fontsize=_g("CH3_LIK_RIDGE_HERE_FS"),
        color="#111111",
        fontfamily="DejaVu Sans",
        fontweight="bold",
        ha=ha,
        va=va,
        zorder=60,
        alpha=a,
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", edgecolor=str(color),
            alpha=0.97 * a,
        ),
    )
    dx, dy = float(px) - lx, float(py) - ly
    d = float(np.hypot(dx, dy)) + 1e-9
    sx = lx + 0.06 * dx / d
    sy = ly + 0.06 * dy / d
    arrow = FancyArrowPatch(
        (sx, sy), (px, py),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.4,
        color=str(color),
        shrinkA=8,
        shrinkB=8,
        zorder=59,
        alpha=a,
    )
    fig.patches.append(arrow)


def _ch6_callout_nearest_side(px, py):
    """Figure edge nearest to projected 3D point."""
    dists = {
        "left": float(px),
        "right": 1.0 - float(px),
        "bottom": float(py),
        "top": 1.0 - float(py),
    }
    return min(dists, key=dists.get)


def _ch6_callout_side_opposite(side):
    return {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}[str(side)]


def _ch6_callout_label_fig_for_side(side, px, py, *, margin=0.07):
    side = str(side)
    if side == "left":
        return (margin, float(np.clip(py, margin + 0.10, 1.0 - margin))), "left", "center"
    if side == "right":
        return ((1.0 - margin), float(np.clip(py, margin + 0.10, 1.0 - margin))), "right", "center"
    if side == "top":
        return (float(np.clip(px, margin + 0.10, 1.0 - margin)), 1.0 - margin), "center", "top"
    return (float(np.clip(px, margin + 0.10, 1.0 - margin)), margin + 0.02), "center", "bottom"


def _ch6_pair_callout_placements(fig, ax3d, pa, pb):
    """Opposite figure edges, each label on the side nearest its point."""
    pa = np.asarray(pa, dtype=np.float64).reshape(3)
    pb = np.asarray(pb, dtype=np.float64).reshape(3)
    fig.canvas.draw()
    pxa, pya = _g("ch3_ax3d_to_fig_xy")(fig, ax3d, float(pa[0]), float(pa[1]), float(pa[2]))
    pxb, pyb = _g("ch3_ax3d_to_fig_xy")(fig, ax3d, float(pb[0]), float(pb[1]), float(pb[2]))
    side_a = _ch6_callout_nearest_side(pxa, pya)
    side_b = _ch6_callout_nearest_side(pxb, pyb)
    if side_a == side_b:
        dists_a = {
            "left": pxa, "right": 1.0 - pxa, "bottom": pya, "top": 1.0 - pya,
        }
        dists_b = {
            "left": pxb, "right": 1.0 - pxb, "bottom": pyb, "top": 1.0 - pyb,
        }
        if dists_a[side_a] <= dists_b[side_b]:
            side_b = _ch6_callout_side_opposite(side_a)
        else:
            side_a = _ch6_callout_side_opposite(side_b)
    fig_a, ha_a, va_a = _ch6_callout_label_fig_for_side(side_a, pxa, pya)
    fig_b, ha_b, va_b = _ch6_callout_label_fig_for_side(side_b, pxb, pyb)
    return (
        {"label_fig": fig_a, "label_ha": ha_a, "label_va": va_a},
        {"label_fig": fig_b, "label_ha": ha_b, "label_va": va_b},
    )


def _ch6_points_within_bounds(pts, bounds):
    """True for each row inside ``bounds`` (dlo1,dhi1,dlo2,dhi2,dlob,dhib)."""
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = (float(v) for v in bounds)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return (
        (pts[:, 0] >= dlo1) & (pts[:, 0] <= dhi1)
        & (pts[:, 1] >= dlo2) & (pts[:, 1] <= dhi2)
        & (pts[:, 2] >= dlob) & (pts[:, 2] <= dhib)
    )


def _ch6_callout_outside_ax3d_placement(fig, ax3d, point, *, edge="top", inset=0.020):
    """Place a callout at the top/bottom rim of the 3D axes (x aligned to point projection)."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    fig.canvas.draw()
    px, _py = _g("ch3_ax3d_to_fig_xy")(fig, ax3d, float(p[0]), float(p[1]), float(p[2]))
    pos = ax3d.get_position()
    lx = float(np.clip(px, pos.x0 + 0.04, pos.x0 + pos.width - 0.04))
    if str(edge) == "top":
        ly = float(np.clip(float(pos.y1) - float(inset), 0.05, 0.97))
        return {
            "label_fig": (lx, ly),
            "label_ha": "center",
            "label_va": "top",
        }
    ly = float(np.clip(float(pos.y0) + float(inset), 0.05, 0.97))
    return {
        "label_fig": (lx, ly),
        "label_ha": "center",
        "label_va": "bottom",
    }


def _ch6_single_callout_placement(fig, ax3d, point, *, margin=0.07):
    """Nearest figure edge to the projected 3D point."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    fig.canvas.draw()
    px, py = _g("ch3_ax3d_to_fig_xy")(fig, ax3d, float(p[0]), float(p[1]), float(p[2]))
    side = _ch6_callout_nearest_side(px, py)
    fig_xy, ha, va = _ch6_callout_label_fig_for_side(side, px, py, margin=margin)
    return {"label_fig": fig_xy, "label_ha": ha, "label_va": va}


def _ch6_callout_placement_for_spec(fig, ax3d, spec):
    anchor = spec.get("label_anchor")
    if anchor:
        if spec.get("label_fig") is not None:
            return {
                "label_fig": tuple(spec["label_fig"]),
                "label_ha": str(spec.get("label_ha", "left")),
                "label_va": str(spec.get("label_va", "top")),
            }
        return _ch6_callout_anchor_placement(fig, ax3d, anchor)
    placement = spec.get("label_placement")
    if placement in ("top_outside", "bottom_outside"):
        edge = "top" if placement == "top_outside" else "bottom"
        return _ch6_callout_outside_ax3d_placement(fig, ax3d, spec["point"], edge=edge)
    if placement in ("left_outside", "right_outside", "diagonal"):
        p = np.asarray(spec["point"], dtype=np.float64).reshape(3)
        fig.canvas.draw()
        px, py = _g("ch3_ax3d_to_fig_xy")(fig, ax3d, float(p[0]), float(p[1]), float(p[2]))
        if placement == "diagonal":
            bounds = spec.get("safe_bounds", _ch6_callout_safe_bounds())
            fig_xy, ha, va = _ch6_diagonal_callout_placement(px, py, bounds)
        else:
            side = "left" if placement == "left_outside" else "right"
            fig_xy, ha, va = _ch6_callout_label_fig_for_side(side, px, py)
        return {"label_fig": fig_xy, "label_ha": ha, "label_va": va}
    if spec.get("pair_opposite_sides") or spec.get("pair_diagonal"):
        return None
    return _ch6_single_callout_placement(fig, ax3d, spec["point"])


def _ch6_voxel_bounds_w12_between_points(pa, pb, base_lim, *, pad_frac=0.15):
    """Voxel view bounds: w_ST/w_EL share span, centered on pair midpoint; b unchanged."""
    pa = np.asarray(pa, dtype=np.float64).reshape(3)
    pb = np.asarray(pb, dtype=np.float64).reshape(3)
    lo_b, hi_b = float(base_lim[0]), float(base_lim[1])
    mid12 = 0.5 * (pa[:2] + pb[:2])
    pad = float(pad_frac) * max(float(np.linalg.norm(pb[:2] - pa[:2])), 1e-6)
    half_st = 0.5 * (max(pa[0], pb[0]) + pad - (min(pa[0], pb[0]) - pad))
    half_el = 0.5 * (max(pa[1], pb[1]) + pad - (min(pa[1], pb[1]) - pad))
    half = max(half_st, half_el, 0.175)

    def _clip_axis(mid, half_w):
        lo, hi = float(mid) - half_w, float(mid) + half_w
        span = hi - lo
        if lo < lo_b:
            lo, hi = lo_b, lo_b + span
        if hi > hi_b:
            hi, lo = hi_b, hi_b - span
        if hi - lo < 0.35:
            mid_c = 0.5 * (lo + hi)
            half_m = 0.175
            lo, hi = max(lo_b, mid_c - half_m), min(hi_b, mid_c + half_m)
        return lo, hi

    lo1, hi1 = _clip_axis(mid12[0], half)
    lo2, hi2 = _clip_axis(mid12[1], half)
    return (lo1, hi1, lo2, hi2, lo_b, hi_b)


def _ch6_voxel_bounds_cube_between_points(pa, pb, base_lim, *, pad_frac=0.15):
    """Like w12 zoom bounds but b shares the same span, centered on the pair midpoint."""
    pa = np.asarray(pa, dtype=np.float64).reshape(3)
    pb = np.asarray(pb, dtype=np.float64).reshape(3)
    lo1, hi1, lo2, hi2, _, _ = _ch6_voxel_bounds_w12_between_points(
        pa, pb, base_lim, pad_frac=pad_frac,
    )
    half = 0.5 * min(float(hi1) - float(lo1), float(hi2) - float(lo2))
    lo_b, hi_b = float(base_lim[0]), float(base_lim[1])
    mid_b = 0.5 * (float(pa[2]) + float(pb[2]))
    lob, hib = mid_b - half, mid_b + half
    span = 2.0 * half
    if lob < lo_b:
        lob, hib = lo_b, lo_b + span
    if hib > hi_b:
        hib, lob = hi_b, hi_b - span
    if hib - lob < 0.35:
        mid_c = 0.5 * (lob + hib)
        half_m = 0.175
        lob, hib = max(lo_b, mid_c - half_m), min(hi_b, mid_c + half_m)
    return (lo1, hi1, lo2, hi2, lob, hib)


def _draw_ch6_param_grad_arrow_3d(
    ax3d, w, grad, *, scale, color="#ef4444", alpha=1.0, ascent=True,
    axis_lim=None, is_direction=False,
):
    """Parameter-space arrow at ``w`` (3-D) with pyramid arrowhead."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    g = np.asarray(grad, dtype=np.float64).reshape(3)
    if is_direction:
        d = g.astype(np.float64)
    else:
        d = (-g if ascent else g).astype(np.float64)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12 or float(scale) < 1e-6 or float(alpha) < 1e-4:
        return
    p1 = w + (d / nrm) * float(scale)
    if axis_lim is None:
        axis_lim = (0.0, max(float(scale) * 3.5, 0.5))
    _ch6_draw_segment(
        ax3d, w, p1,
        color=str(color), lw=2.6, alpha=float(alpha),
        arrowhead=True, axis_lim=axis_lim,
    )


def _ch6_newton_dir_w12(w, Xd, y, *, ridge=CH6_RIDGE, H_scale=1.0):
    """Newton push direction in (w_ST, w_EL): −H⁻¹∇NLL (toward the bowl minimum)."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    g = ch6_nll_grad(w, Xd, y, ridge=ridge)[:2].astype(np.float64)
    H = (
        float(H_scale)
        * ch6_observed_information(w, Xd, y, ridge=ridge)[:2, :2].astype(np.float64)
    )
    H = H + 0.10 * np.eye(2, dtype=np.float64)
    try:
        d = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        gn = float(np.linalg.norm(g))
        d = -g if gn > 1e-12 else np.zeros(2, dtype=np.float64)
    return d.astype(np.float64)


def _ch6_newton_ascent_dir(w, Xd, y, *, ridge=CH6_RIDGE):
    """Unit Newton / Hessian-informed ascent direction −H⁻¹∇NLL at ``w``."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    g = ch6_nll_grad(w, Xd, y, ridge=ridge)
    H = ch6_observed_information(w, Xd, y, ridge=ridge) + 0.10 * np.eye(3, dtype=np.float64)
    try:
        d = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        gn = float(np.linalg.norm(g))
        d = -g / gn if gn > 1e-12 else np.zeros(3, dtype=np.float64)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return d / nrm


def _ch6_lerp_axis_lim(a, b, u):
    u = float(np.clip(float(u), 0.0, 1.0))
    return (
        float(a[0]) + u * (float(b[0]) - float(a[0])),
        float(a[1]) + u * (float(b[1]) - float(a[1])),
    )


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


def _draw_inward_gaussian_surface(
    ax3d,
    W,
    *,
    plane_i,
    plane_j,
    height_k,
    axis_lim,
    side=None,
    reveal_u=1.0,
    grid_n=72,
):
    """Smooth Gaussian surface on one coordinate plane, extruded inward."""
    u = float(np.clip(reveal_u, 0.0, 1.0))
    if u <= 1e-6:
        return
    W = np.asarray(W, dtype=np.float64)
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    side = int(side) if side is not None else _inward_hist_side(W, height_k, axis_lim)
    uv = np.column_stack([W[:, plane_i], W[:, plane_j]])
    mu = np.mean(uv, axis=0)
    C = np.cov(uv.T)
    C = np.asarray(C, dtype=np.float64) + 1e-6 * np.eye(2, dtype=np.float64)
    Ci = np.linalg.inv(C)

    xs = np.linspace(lo, hi, int(grid_n))
    ys = np.linspace(lo, hi, int(grid_n))
    X, Y = np.meshgrid(xs, ys)
    D = np.stack([X - mu[0], Y - mu[1]], axis=-1)
    E = np.einsum("...i,ij,...j->...", D, Ci, D)
    Zg = np.exp(-0.5 * E)
    Zg /= max(float(np.max(Zg)), 1e-8)

    mu_k = float(np.mean(W[:, height_k]))
    if side < 0:
        k_plane = lo
        peak_h = 0.5 * max(mu_k - k_plane, 1e-6)
        Zk = k_plane + Zg * peak_h * u
    else:
        k_plane = hi
        peak_h = 0.5 * max(k_plane - mu_k, 1e-6)
        Zk = k_plane - Zg * peak_h * u

    X3 = np.zeros_like(X, dtype=np.float64)
    Y3 = np.zeros_like(Y, dtype=np.float64)
    Z3 = np.zeros_like(Zk, dtype=np.float64)
    coords = [X3, Y3, Z3]
    coords[plane_i][:] = X
    coords[plane_j][:] = Y
    coords[height_k][:] = Zk

    rgba = _ch6_density_belief_rgba(
        Zg.reshape(-1), z_lim=(0.0, 1.0), alpha=0.90 * u,
    ).reshape(*Zg.shape, 4)
    ax3d.plot_surface(
        X3, Y3, Z3,
        facecolors=rgba,
        rstride=1, cstride=1,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        zorder=4,
    )


def _draw_gaussian_voxel_ellipsoid(
    ax3d,
    mean,
    cov,
    *,
    axis_lim,
    mass=0.95,
    reveal_u=1.0,
    n_cells=20,
):
    """Voxelized Gaussian region (credible ellipsoid) in (w_ST, w_EL, b)."""
    from scipy.stats import chi2

    u = float(np.clip(reveal_u, 0.0, 1.0))
    if u <= 1e-6:
        return
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    n = int(max(8, n_cells))
    w1e = np.linspace(lo, hi, n + 1, dtype=np.float64)
    w2e = np.linspace(lo, hi, n + 1, dtype=np.float64)
    be = np.linspace(lo, hi, n + 1, dtype=np.float64)
    w1c = 0.5 * (w1e[:-1] + w1e[1:])
    w2c = 0.5 * (w2e[:-1] + w2e[1:])
    bc = 0.5 * (be[:-1] + be[1:])

    W1, W2, B = np.meshgrid(w1c, w2c, bc, indexing="ij")
    mu = np.asarray(mean, dtype=np.float64).reshape(3)
    C = np.asarray(cov, dtype=np.float64).reshape(3, 3) + 1e-6 * np.eye(3, dtype=np.float64)
    Ci = np.linalg.inv(C)
    D = np.stack([W1 - mu[0], W2 - mu[1], B - mu[2]], axis=-1)
    E = np.einsum("...i,ij,...j->...", D, Ci, D)
    r2 = float(chi2.ppf(float(mass), 3))
    # Grow from center outward during reveal.
    active = E <= (u * r2 + 1e-9)
    if not np.any(active):
        return
    dens = np.exp(-0.5 * E)
    dmax = max(float(np.max(dens[active])), 1e-9)
    t = np.zeros_like(dens)
    t[active] = dens[active] / dmax
    fc = np.zeros(active.shape + (4,), dtype=np.float64)
    if np.any(active):
        rgba = _ch6_density_belief_rgba(
            t[active].reshape(-1), z_lim=(0.0, 1.0), alpha=0.78,
        )
        fc[active] = rgba
    ax3d.voxels(
        w1e, w2e, be,
        active,
        facecolors=fc,
        edgecolor="#0f172a",
        linewidth=0.12,
        shade=False,
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
# Signed-difference palette — readable on belief pink/blue; mix reads as lime.
CH6_VARIANCE_POS_COLOR = "#42A5F5"
CH6_VARIANCE_NEG_COLOR = "#FFEA00"
CH6_VARIANCE_MIX_COLOR = "#9ccc65"
CH6_HESSIAN_ARROW_COLOR = "#111111"
CH6_COV_PLANE_ELEV = 90.0
CH6_COV_PLANE_AZIM = -90.0
CH6_COV_PLANE_ZOOM = 3.05
CH6_COV_WST_VIEW = (0.0, 2.5)
CH6_COV_WEL_VIEW = (-2.5, 0.0)
CH6_COV_EQ_COLOR = "#111111"
CH6_COV_AXIS_SOLID_LW = 4.8
CH6_COV_AXIS_DOTTED_LW = 6.2
CH6_COV_RANGE_LW = 4.2
CH6_COV_MATH_FS = 22.5
CH6_VARIANCE_MATH_FS = 28.0
CH6_VARIANCE_DOTTED_LW = 3.0
CH6_VARIANCE_MATH_BOX_PAD_TOP_PT = 10.0
CH6_VARIANCE_MATH_BOX_PAD_BOTTOM_PT = 5.0
CH6_MU_WST_TEX = r"\mu_{w_{\mathrm{ST}}}"
CH6_MU_WEL_TEX = r"\mu_{w_{\mathrm{EL}}}"
CH6_VAR_MEAN_ZERO_CAPTION = r"The average deviation is zero"
CH6_COV_PANEL_BOX_X = 0.02
CH6_COV_PANEL_BOX_W = 0.96
CH6_COV_PANEL_BOX_Y = 0.04
CH6_COV_PANEL_BOX_H = 0.92
CH6_COV_PANEL_BOX_LW = 2.8
CH6_COV_BEAT_CAPTION_FS_SCALE = 1.12
CH6_COV_BEAT_CAPTION_UP_CM = 2.0
CH6_COV_DEMO_MARKER_ZORDER = 22
CH6_COV_DEMO_MARKER_SCALE = 2.15
CH6_COV_BLOCK_H_MAX = 0.095
CH6_COV_BEAT_CAPTIONS = (
    r"$w_{\mathrm{EL}}$ and $w_{\mathrm{ST}}$ vary mostly in opposite ways",
    r"$w_{\mathrm{EL}}$ and $w_{\mathrm{ST}}$ vary mostly in the same way",
    r"$w_{\mathrm{EL}}$ and $w_{\mathrm{ST}}$ don't seem to vary together in a consistent way",
)
CH6_COV_MISMATCH_COLOR = "#AB47BC"
CH6_COV_PRODUCT_NEG_COLOR = "#AB47BC"
CH6_COV_PRODUCT_POS_COLOR = "#42A5F5"
# Coordinate planes for combined Hessian directions (w_st=0, w_el=1, b=2).
CH6_HESSIAN_PLANE_AXES: tuple[tuple[int, int], ...] = (
    (0, 2),  # w_st – b
    (0, 1),  # w_st – w_el
    (1, 2),  # b – w_el  (w_el – b)
)
CH6_VARIANCE_AXIS_NAMES = ("w_st", "w_el", "b")
CH6_MATRIX_AXIS_LABELS = (
    r"$w_{\mathrm{ST}}$",
    r"$w_{\mathrm{EL}}$",
    r"$b$",
)
CH6_MATRIX_DECOMP_COLORS = ("#42A5F5", "#FF9800", "#7E57C2")
CH6_MATRIX_VAR_COLOR = "#AB47BC"
CH6_MATRIX_COV_TILT_COLORS = ("#7B1FA2", "#4A148C", "#311B92")
CH6_MATRIX_HIGHLIGHT_COLOR = "#AB47BC"
CH6_MATRIX_VAR_HIGHLIGHT_FACE = "#fefcff"
CH6_MATRIX_VAR_HIGHLIGHT_EDGE = "#e4d9fc"
CH6_MATRIX_VAR_EXPLORED_FACE = "#f3ecff"
CH6_MATRIX_VAR_EXPLORED_EDGE = "#e6daff"
CH6_MATRIX_COV_HIGHLIGHT_FACE = "#ede9fe"
CH6_MATRIX_COV_EXPLORED_FACE = "#e4d8ff"
CH6_MATRIX_COV_EXPLORED_EDGE = "#c4b5fd"
CH6_MATRIX_ARROW_SCALE_FRAC = 0.38
CH6_MATRIX_ARROW_BOOST = 2.4
CH6_MATRIX_ARROW_HEAD_LEN_FRAC = 0.050
CH6_MATRIX_ARROW_HEAD_HALF_FRAC = 0.36
CH6_MATRIX_PANEL_TITLE_FS = 16.0
CH6_MATRIX_PANEL_LABEL_FS = 14.0
CH6_MATRIX_PANEL_CELL_FS = 13.0
CH6_MATRIX_PANEL_ZERO_FS = 14.0
CH6_MATRIX_PANEL_TITLE_Y = 1.02
CH6_MATRIX_TRIPTYCH_LEFT = (0.02, 0.10, 0.24, 0.80)
CH6_MATRIX_TRIPTYCH_CENTER = (0.28, 0.06, 0.44, 0.88)
CH6_MATRIX_TRIPTYCH_RIGHT = (0.74, 0.10, 0.24, 0.80)
CH6_NLL_VOXEL_ALPHA_HI = 0.92
CH6_NLL_VOXEL_ALPHA_LO = 0.02
CH6_NLL_VOXEL_CELL_MULT_DRAFT = 4
CH6_NLL_VOXEL_CELL_MULT_HQ = 6
CH6_NLL_VOXEL_GAP_PITCH_DRAFT = 1
CH6_NLL_VOXEL_GAP_PITCH_HQ = 1
CH6_NLL_VOXEL_CHECKER_FREQ = 0.125
CH6_NLL_VOXEL_USE_CHECKER = True
CH6_NLL_VOXEL_CURV_REF_PCT = 30.0
CH6_NLL_VOXEL_CURV_REF_PCT_MIN = 1.0
CH6_NLL_VOXEL_CURV_REF_PCT_MAX = 100.0
CH6_NLL_VOXEL_WALKER_CLEAR_REF_PCT = 12.0
CH6_NLL_VOXEL_WALKER_CLEAR_MAX = 0.72
CH6_NLL_VOXEL_WALKER_OPAQUE_MAX = 0.92
CH6_VOXEL_MEDIUM_UNIFORM_ALPHA = 0.11
CH6_VOXEL_MEDIUM_WAKE_ALPHA = 0.50
CH6_VOXEL_MEDIUM_DEMO_AXIS_LIM = (-1.0, 1.0)
CH6_VOXEL_MEDIUM_DEMO_W_START = (-0.58, -0.50, -0.72)
CH6_VOXEL_MEDIUM_DEMO_W_TARGET = (0.62, 0.54, 0.76)
CH6_VOXEL_MEDIUM_SYNTH_HESSIAN = np.diag([14.0, 14.0, 9.0])
CH6_VOXEL_MEDIUM_PUSH_ETA = 0.062
CH6_VOXEL_MEDIUM_CURV_FLAT = 0.085
CH6_VOXEL_MEDIUM_CURV_MEDIUM = 1.05
CH6_VOXEL_MEDIUM_CURV_STEEP = 6.8
CH6_NLL_VOXEL_ZOOM_MAX_AXIS_CELLS_DRAFT = 18
CH6_NLL_VOXEL_ZOOM_MAX_AXIS_CELLS_HQ = 24
CH6_NLL_VOXEL_TRAIL_SUBDIV_DRAFT = 48
CH6_NLL_VOXEL_TRAIL_SUBDIV_HQ = 128
CH6_NLL_VOXEL_TRAIL_HALF_CELLS_DRAFT = 5
CH6_NLL_VOXEL_TRAIL_HALF_CELLS_HQ = 12
CH6_NLL_VOXEL_SWEEP_WAVE_AMP_DRAFT = 0.30
CH6_NLL_VOXEL_SWEEP_WAVE_AMP_HQ = 0.42
CH6_NLL_VOXEL_TRAIL_CHECKER_FREQ = 0.125
CH6_SPRING_ZOOM_BIAS_BOUNDS = (-1.5, 1.5)
CH6_W12_PUSH_SURFACE_ALPHA = 0.24
CH6_W12_PUSH_MARKER_S = 148.0
CH6_W12_PUSH_GRAD_SCALE = 0.40
CH6_W12_PUSH_LOW_CURV_MULT = 0.55
CH6_W12_PUSH_HIGH_CURV_MULT = 3.6
CH6_W12_PUSH_STEP_ETA = 0.072
CH6_VARIANCE_VIEWS: dict[str, tuple[float, float]] = {
    # elev=0; cumulative +90° azim per pass (bias → w_ST → w_EL)
    "b": (0.0, 0.0),
    "w_st_face": (0.0, 90.0),
    "w_el_face": (0.0, 180.0),
}
CH6_VARIANCE_INTERVAL_OFFSET = 0.10


def _ch6_nll_voxel_cell_mult():
    from ch6_frequentist import _CH3_DRAFT

    return int(CH6_NLL_VOXEL_CELL_MULT_DRAFT if _CH3_DRAFT else CH6_NLL_VOXEL_CELL_MULT_HQ)


def _ch6_nll_voxel_gap_pitch():
    from ch6_frequentist import _CH3_DRAFT

    return int(CH6_NLL_VOXEL_GAP_PITCH_DRAFT if _CH3_DRAFT else CH6_NLL_VOXEL_GAP_PITCH_HQ)


def _ch6_nll_voxel_zoom_max_axis_cells():
    from ch6_frequentist import _CH3_DRAFT

    return int(
        CH6_NLL_VOXEL_ZOOM_MAX_AXIS_CELLS_DRAFT
        if _CH3_DRAFT
        else CH6_NLL_VOXEL_ZOOM_MAX_AXIS_CELLS_HQ
    )


def _ch6_nll_voxel_sweep_wave_amp():
    from ch6_frequentist import _CH3_DRAFT

    return float(
        CH6_NLL_VOXEL_SWEEP_WAVE_AMP_DRAFT
        if _CH3_DRAFT
        else CH6_NLL_VOXEL_SWEEP_WAVE_AMP_HQ
    )


def _ch6_nll_voxel_reference_cell_size(reference_bounds, *, cell_mult=None):
    """Edge length of one voxel cube matching ch6_127 full-axis density."""
    dlo1, dhi1, _, _, _, _ = (float(v) for v in reference_bounds)
    span_ref = float(dhi1 - dlo1)
    mult = int(_ch6_nll_voxel_cell_mult() if cell_mult is None else cell_mult)
    n_ref = int(_g("_ch4_voxel_n_cells")(mult))
    return span_ref / max(n_ref, 2)


def _ch6_nll_voxel_curv_quad(W1, W2, B, mu, H):
    """Curvature-weighted squared distance δᵀ H δ on the voxel grid."""
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    H = 0.5 * (H + H.T)
    evals, evecs = np.linalg.eigh(H)
    evals = np.clip(evals, 1e-12, None)
    delta = np.stack(
        [W1 - mu[0], W2 - mu[1], B - mu[2]],
        axis=-1,
    )
    coeff = delta @ evecs
    return np.sum(evals * coeff * coeff, axis=-1)


def _ch6_nll_voxel_alpha_from_quad(quad, ref_pct):
    ref_pct = float(np.clip(ref_pct, 1.0, 100.0))
    ref = float(np.nanpercentile(quad, ref_pct))
    ref = max(ref, 1e-9)
    rel = np.clip(quad / ref, 0.0, 1.0)
    return (
        float(CH6_NLL_VOXEL_ALPHA_HI) * (1.0 - rel)
        + float(CH6_NLL_VOXEL_ALPHA_LO) * rel
    )


def _ch6_nll_voxel_curv_ref_pct_between(u, lo, hi):
    """Smoothly interpolate curvature ref percentile from lo → hi over u ∈ [0, 1]."""
    u = float(_g("ch3_knob_smoothstep")(float(np.clip(float(u), 0.0, 1.0))))
    lo = float(lo)
    hi = float(hi)
    return lo + (hi - lo) * u


def _ch6_nll_voxel_alpha_curvature(W1, W2, B, mu, H, *, ref_pct=None):
    """Transparency from μ with per-direction decay ∝ Hessian curvature (δᵀ H δ)."""
    if ref_pct is None:
        ref_pct = float(CH6_NLL_VOXEL_CURV_REF_PCT)
    quad = _ch6_nll_voxel_curv_quad(W1, W2, B, mu, H)
    return _ch6_nll_voxel_alpha_from_quad(quad, ref_pct)


def _ch6_nll_voxel_alpha_linear(nll, *, lo, hi):
    """Linear falloff: low NLL (good) opaque, high NLL (bad) faint."""
    nll = np.asarray(nll, dtype=np.float64)
    lo = float(lo)
    hi = float(hi)
    span = max(hi - lo, 1e-9)
    rel = np.clip((nll - lo) / span, 0.0, 1.0)
    return (
        float(CH6_NLL_VOXEL_ALPHA_HI) * (1.0 - rel)
        + float(CH6_NLL_VOXEL_ALPHA_LO) * rel
    )


def _ch6_nll_voxel_checker_mask(n, *, gap_pitch, freq=1.0):
    """Regular-spaced voxel lattice; ``freq`` selects stride along each axis."""
    n = int(n)
    freq = float(np.clip(freq, 0.0, 1.0))
    if freq >= 1.0 - 1e-9:
        return _g("_ch4_voxel_checker_mask")(n, gap_pitch=int(gap_pitch))
    if freq >= 0.5 - 1e-9:
        return _g("_ch4_voxel_checker_mask")(n, gap_pitch=2)
    ii, jj, kk = np.indices((n, n, n))
    stride = 2 if freq >= 0.25 - 1e-9 else 3
    return (ii % stride == 0) & (jj % stride == 0) & (kk % stride == 0)


def _ch6_nll_voxel_checker_mask_aniso(n1, n2, n3, *, gap_pitch, freq=1.0):
    """Checker mask for non-cubic voxel grids."""
    n1, n2, n3 = int(n1), int(n2), int(n3)
    if n1 == n2 == n3:
        return _ch6_nll_voxel_checker_mask(n1, gap_pitch=int(gap_pitch), freq=float(freq))
    freq = float(np.clip(freq, 0.0, 1.0))
    ii, jj, kk = np.indices((n1, n2, n3))
    if freq >= 1.0 - 1e-9:
        pitch = max(1, int(gap_pitch))
        return ((ii // pitch) + (jj // pitch) + (kk // pitch)) % 2 == 0
    stride = 2 if freq >= 0.5 - 1e-9 else (2 if freq >= 0.25 - 1e-9 else 3)
    return (ii % stride == 0) & (jj % stride == 0) & (kk % stride == 0)


def _ch6_nll_voxel_cache(
    study,
    exam,
    y,
    *,
    bounds=None,
    cell_mult=None,
    gap_pitch=None,
    cubic_cells=False,
    cell_mult_boost=0,
    max_axis_cells=None,
    reference_bounds=None,
):
    """NLL checkerboard voxels (Ch4 colors; curvature-weighted transparency from μ)."""
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if bounds is None:
        bounds = _g("CH3_LIK_CT_VIEW_BOUNDS")
    cell_mult = int(_ch6_nll_voxel_cell_mult() if cell_mult is None else cell_mult)
    cell_mult = cell_mult + int(cell_mult_boost)
    gap_pitch = int(_ch6_nll_voxel_gap_pitch() if gap_pitch is None else gap_pitch)
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = tuple(float(v) for v in bounds)
    spans = (dhi1 - dlo1, dhi2 - dlo2, dhib - dlob)
    if cubic_cells:
        if reference_bounds is not None:
            cell_size = _ch6_nll_voxel_reference_cell_size(
                reference_bounds, cell_mult=cell_mult,
            )
        else:
            n_ref = int(_g("_ch4_voxel_n_cells")(cell_mult))
            cell_size = min(spans) / max(n_ref, 4)
        n1 = max(2, int(round(spans[0] / cell_size)))
        n2 = max(2, int(round(spans[1] / cell_size)))
        n3 = max(2, int(round(spans[2] / cell_size)))
        if max_axis_cells is not None:
            cap = max(4, int(max_axis_cells))
            peak = max(n1, n2, n3)
            if peak > cap:
                scale = float(peak) / float(cap)
                n1 = max(4, int(round(n1 / scale)))
                n2 = max(4, int(round(n2 / scale)))
                n3 = max(4, int(round(n3 / scale)))
    else:
        n1 = n2 = n3 = int(_g("_ch4_voxel_n_cells")(cell_mult))
    x_edges = _g("_ch4_voxel_cell_edges")(dlo1, dhi1, n1)
    y_edges = _g("_ch4_voxel_cell_edges")(dlo2, dhi2, n2)
    z_edges = _g("_ch4_voxel_cell_edges")(dlob, dhib, n3)
    w1 = 0.5 * (x_edges[:-1] + x_edges[1:])
    w2 = 0.5 * (y_edges[:-1] + y_edges[1:])
    bb = 0.5 * (z_edges[:-1] + z_edges[1:])
    W1, W2, B = np.meshgrid(w1, w2, bb, indexing="ij")
    nll = _g("_ch3_nll_sum_on_flat_grid")(
        study, exam, y,
        W1.ravel(), W2.ravel(), B.ravel(),
    ).reshape(W1.shape)
    g_lo, g_hi = _g("ch4_nll_global_scale")()
    w_hat, _ = ch6_fit_dataset(study, exam, y, ridge=CH6_RIDGE)
    mu = np.asarray(w_hat, dtype=np.float64).reshape(3)
    H = ch6_observed_information(mu, ch6_design(study, exam), y, ridge=CH6_RIDGE)
    rgba = np.asarray(
        ch4_nll_heatmap_facecolors(nll, vmin=g_lo, vmax=g_hi, alpha=1.0),
        dtype=np.float32,
    )
    curv_quad = _ch6_nll_voxel_curv_quad(W1, W2, B, mu, H).astype(np.float32, copy=False)
    if CH6_NLL_VOXEL_USE_CHECKER:
        checker = _ch6_nll_voxel_checker_mask_aniso(
            n1, n2, n3,
            gap_pitch=int(gap_pitch), freq=float(CH6_NLL_VOXEL_CHECKER_FREQ),
        )
    else:
        checker = np.ones((int(n1), int(n2), int(n3)), dtype=bool)
    start = np.array(_g("CH3_LIK_VOXEL_DIAG_START"), dtype=np.float64)
    end = np.array(_g("CH3_LIK_VOXEL_DIAG_END"), dtype=np.float64)
    diag = end - start
    max_proj = float(np.dot(diag, diag))
    proj = (
        (W1 - start[0]) * diag[0]
        + (W2 - start[1]) * diag[1]
        + (B - start[2]) * diag[2]
    ).astype(np.float32, copy=False)
    return {
        "n": (n1, n2, n3),
        "x_edges": x_edges,
        "y_edges": y_edges,
        "z_edges": z_edges,
        "checker": checker,
        "proj": proj,
        "max_proj": max_proj,
        "rgb": rgba[..., :3],
        "curv_quad": curv_quad,
        "mu": mu,
        "H": H,
        "bounds": (dlo1, dhi1, dlo2, dhi2, dlob, dhib),
    }


def _ch6_landscape_push_axes_2d(w0, w_goal):
    """Unit tangent/normal to the push in w_ST–w_EL."""
    w0 = np.asarray(w0, dtype=np.float64).reshape(3)
    w_goal = np.asarray(w_goal, dtype=np.float64).reshape(3)
    u = w_goal[:2] - w0[:2]
    un = float(np.linalg.norm(u))
    if un < 1e-9:
        u = np.array([1.0, 0.0], dtype=np.float64)
    else:
        u = u / un
    v = np.array([-u[1], u[0]], dtype=np.float64)
    return u, v, un


def _ch6_landscape_push_hessian(w0, w_goal, kind):
    """3×3 Hessian for the push integrator (curvature felt along the walk)."""
    u, v, _ = _ch6_landscape_push_axes_2d(w0, w_goal)
    if kind == "flat_plateau":
        H2 = 0.028 * np.eye(2, dtype=np.float64)
    elif kind == "steep_decay":
        H2 = 85.0 * np.outer(u, u) + 190.0 * np.outer(v, v)
    elif kind == "immobile_valley":
        H2 = 2000.0 * np.outer(u, u) + 2600.0 * np.outer(v, v)
    else:
        H2 = 0.028 * np.eye(2, dtype=np.float64)
    H = np.zeros((3, 3), dtype=np.float64)
    H[:2, :2] = H2
    H[2, 2] = 0.08
    return H


def _ch6_landscape_pocket_r2_cells(W1, W2, B, pt, bounds):
    """Squared distance in units of one voxel edge (≈1 cell at the pocket center)."""
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = (float(v) for v in bounds)
    n1 = int(W1.shape[0])
    n2 = int(W1.shape[1])
    n3 = int(W1.shape[2])
    sx = max((dhi1 - dlo1) / max(n1, 1), 1e-6)
    sy = max((dhi2 - dlo2) / max(n2, 1), 1e-6)
    sz = max((dhib - dlob) / max(n3, 1), 1e-6)
    pt = np.asarray(pt, dtype=np.float64).reshape(3)
    return (
        ((W1 - pt[0]) / sx) ** 2
        + ((W2 - pt[1]) / sy) ** 2
        + ((B - pt[2]) / sz) ** 2
    )


def _ch6_landscape_stamp_pocket_floor(obj, W1, W2, B, pt, floor, *, bounds=None, radius_cells=2.4):
    """Pull a small 3-D cluster of lattice sites down to ``floor``."""
    obj = np.asarray(obj, dtype=np.float64).copy()
    pt = np.asarray(pt, dtype=np.float64).reshape(3)
    if bounds is not None:
        r2 = _ch6_landscape_pocket_r2_cells(W1, W2, B, pt, bounds)
        mask = r2 <= float(radius_cells) * float(radius_cells)
    else:
        r2 = (W1 - pt[0]) ** 2 + (W2 - pt[1]) ** 2 + (B - pt[2]) ** 2
        mask = r2 <= (0.08 * 0.08)
    obj[mask] = np.minimum(obj[mask], float(floor))
    return obj


def _ch6_landscape_nll_grid(W1, W2, B, w0, w_min, kind, *, w_dest=None, bounds=None):
    """Objective to minimize: dark at ``w_min``, mostly pink elsewhere."""
    w0 = np.asarray(w0, dtype=np.float64).reshape(3)
    w_min = np.asarray(w_min, dtype=np.float64).reshape(3)
    u, v, span = _ch6_landscape_push_axes_2d(w0, w_min)
    span = max(span, 0.25)
    d0 = W1 - w0[0]
    d1 = W2 - w0[1]
    along = d0 * u[0] + d1 * u[1]
    cross = d0 * v[0] + d1 * v[1]
    am = (W1 - w_min[0]) * u[0] + (W2 - w_min[1]) * u[1]
    cm = (W1 - w_min[0]) * v[0] + (W2 - w_min[1]) * v[1]
    dm = B - w_min[2]
    r2_start = along * along + cross * cross + 0.30 * (B - w0[2]) ** 2
    r2_min = am * am + cm * cm + 0.30 * dm * dm
    if kind == "flat_plateau":
        obj = 1.06 + 0.016 * (1.0 - np.exp(-r2_min / max((0.55 * span) ** 2, 1e-9)))
    elif kind == "steep_decay":
        obj = (
            1.05
            + 0.26 * (am / span) * (am / span)
            + 34.0 * cm * cm
            + 3.6 * np.maximum(am, 0.0) ** 2 / (span * span)
            + 0.10 * dm * dm
        )
    elif kind == "immobile_valley":
        w_dest_pt = np.asarray(
            w_min if w_dest is None else w_dest, dtype=np.float64,
        ).reshape(3)
        w_pocket = w_dest_pt.copy()
        if bounds is None:
            r2 = (
                (W1 - w_pocket[0]) ** 2
                + (W2 - w_pocket[1]) ** 2
                + 0.65 * (B - w_pocket[2]) ** 2
            )
            pocket_radius_cells = 0.095
        else:
            dlo1, dhi1, dlo2, dhi2, dlob, dhib = (float(v) for v in bounds)
            n_ref = max(int(W1.shape[0]), 1)
            sx = max((dhi1 - dlo1) / n_ref, 1e-6)
            sy = max((dhi2 - dlo2) / n_ref, 1e-6)
            sz = max((dhib - dlob) / n_ref, 1e-6)
            d_mu = w0 - w_pocket
            reach_mu = float(
                np.sqrt(
                    (d_mu[0] / sx) ** 2
                    + (d_mu[1] / sy) ** 2
                    + 0.65 * (d_mu[2] / sz) ** 2
                )
            )
            pocket_radius_cells = max(1.55 * reach_mu + 2.35, 2.85) * 5.0
            r2 = _ch6_landscape_pocket_r2_cells(W1, W2, B, w_pocket, bounds)
        r_cells = np.sqrt(np.maximum(r2, 0.0))
        t = np.clip(r_cells / float(pocket_radius_cells), 0.0, 1.0)
        bump = 0.5 * (1.0 + np.cos(np.pi * t))
        pink = 1.14
        depth = 6.2
        obj = pink - depth * bump
    else:
        obj = 0.5 * r2_min
    return np.asarray(obj, dtype=np.float64)


def _ch6_landscape_voxel_color_limits(obj, kind):
    """Colormap limits: dark = low objective at ``w_min``."""
    arr = np.asarray(obj, dtype=np.float64)
    if kind == "flat_plateau":
        g_lo = float(np.nanmin(arr))
        g_hi = g_lo + 0.028
    elif kind == "steep_decay":
        g_lo = float(np.nanpercentile(arr, 4.0))
        g_hi = float(np.nanpercentile(arr, 62.0))
    elif kind == "immobile_valley":
        g_lo = float(np.nanmin(arr))
        g_hi = 1.14
    else:
        g_lo = float(np.nanpercentile(arr, 6.0))
        g_hi = float(np.nanpercentile(arr, 94.0))
    if g_hi <= g_lo + 1e-9:
        g_hi = g_lo + 1.0
    return g_lo, g_hi


def _ch6_clip_w_to_bounds(w, bounds):
    w = np.asarray(w, dtype=np.float64).reshape(3)
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = (float(v) for v in bounds)
    return np.array([
        float(np.clip(w[0], dlo1, dhi1)),
        float(np.clip(w[1], dlo2, dhi2)),
        float(np.clip(w[2], dlob, dhib)),
    ], dtype=np.float64)


def _ch6_chord_goal_in_bounds(w0, w_goal, bounds):
    """Furthest point from ``w0`` toward ``w_goal`` that lies inside ``bounds``."""
    w0 = np.asarray(w0, dtype=np.float64).reshape(3)
    w_goal = np.asarray(w_goal, dtype=np.float64).reshape(3)
    if bool(_ch6_points_within_bounds(w_goal, bounds)[0]):
        return _ch6_clip_w_to_bounds(w_goal, bounds)
    delta = w_goal - w0
    for t in np.linspace(1.0, 0.0, 80):
        pt = w0 + float(t) * delta
        if bool(_ch6_points_within_bounds(pt, bounds)[0]):
            return _ch6_clip_w_to_bounds(pt, bounds)
    return _ch6_clip_w_to_bounds(w0, bounds)


def _ch6_immobile_near_dest(w0, w_push, bounds, *, frac=0.042):
    """Place the black marker a short step from μ so both sit in one pocket."""
    w0 = np.asarray(w0, dtype=np.float64).reshape(3)
    w_push = np.asarray(w_push, dtype=np.float64).reshape(3)
    delta = w_push - w0
    gn = float(np.linalg.norm(delta))
    if gn < 1e-12:
        return _ch6_clip_w_to_bounds(w0, bounds)
    w_near = w0 + float(frac) * delta
    return _ch6_clip_w_to_bounds(w_near, bounds)


def _ch6_clip_path_to_bounds(path_3d, bounds):
    return [_ch6_clip_w_to_bounds(w, bounds) for w in path_3d]


def _ch6_landscape_pocket_checker_mask(W1, W2, B, pockets, bounds, *, radius_cells=2.6):
    """Sparse checker plus guaranteed sites around each pocket center."""
    n1, n2, n3 = W1.shape
    if CH6_NLL_VOXEL_USE_CHECKER:
        checker = _ch6_nll_voxel_checker_mask_aniso(
            int(n1), int(n2), int(n3),
            gap_pitch=int(_ch6_nll_voxel_gap_pitch()),
            freq=float(CH6_NLL_VOXEL_CHECKER_FREQ),
        )
    else:
        checker = np.ones((int(n1), int(n2), int(n3)), dtype=bool)
    rad2 = float(radius_cells) * float(radius_cells)
    for pt in pockets:
        r2 = _ch6_landscape_pocket_r2_cells(W1, W2, B, pt, bounds)
        checker |= r2 <= rad2
    return checker


def _ch6_landscape_voxel_cache(bounds, w0, w_min, kind, *, H_push=None, w_dest=None):
    """Voxel lattice colored by a named synthetic landscape."""
    w0 = np.asarray(w0, dtype=np.float64).reshape(3)
    w_min = np.asarray(w_min, dtype=np.float64).reshape(3)
    if H_push is None:
        H_push = _ch6_landscape_push_hessian(w0, w_min, kind)
    H_push = np.asarray(H_push, dtype=np.float64).reshape(3, 3)
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = tuple(float(v) for v in bounds)
    cell_mult = int(_ch6_nll_voxel_cell_mult())
    gap_pitch = int(_ch6_nll_voxel_gap_pitch())
    n1 = n2 = n3 = int(_g("_ch4_voxel_n_cells")(cell_mult))
    x_edges = _g("_ch4_voxel_cell_edges")(dlo1, dhi1, n1)
    y_edges = _g("_ch4_voxel_cell_edges")(dlo2, dhi2, n2)
    z_edges = _g("_ch4_voxel_cell_edges")(dlob, dhib, n3)
    w1 = 0.5 * (x_edges[:-1] + x_edges[1:])
    w2 = 0.5 * (y_edges[:-1] + y_edges[1:])
    bb = 0.5 * (z_edges[:-1] + z_edges[1:])
    W1, W2, B = np.meshgrid(w1, w2, bb, indexing="ij")
    nll = _ch6_landscape_nll_grid(
        W1, W2, B, w0, w_min, kind, w_dest=w_dest, bounds=bounds,
    )
    g_lo, g_hi = _ch6_landscape_voxel_color_limits(nll, kind)
    rgba = np.asarray(
        ch4_nll_heatmap_facecolors(nll, vmin=g_lo, vmax=g_hi, alpha=1.0),
        dtype=np.float32,
    )
    curv_quad = _ch6_nll_voxel_curv_quad(W1, W2, B, w0, H_push).astype(np.float32, copy=False)
    if CH6_NLL_VOXEL_USE_CHECKER:
        checker = _ch6_nll_voxel_checker_mask_aniso(
            int(n1), int(n2), int(n3),
            gap_pitch=int(gap_pitch), freq=float(CH6_NLL_VOXEL_CHECKER_FREQ),
        )
    else:
        checker = np.ones((int(n1), int(n2), int(n3)), dtype=bool)
    start = np.array(_g("CH3_LIK_VOXEL_DIAG_START"), dtype=np.float64)
    end = np.array(_g("CH3_LIK_VOXEL_DIAG_END"), dtype=np.float64)
    diag = end - start
    max_proj = float(np.dot(diag, diag))
    proj = (
        (W1 - start[0]) * diag[0]
        + (W2 - start[1]) * diag[1]
        + (B - start[2]) * diag[2]
    ).astype(np.float32, copy=False)
    return {
        "n": (n1, n2, n3),
        "x_edges": x_edges,
        "y_edges": y_edges,
        "z_edges": z_edges,
        "checker": checker,
        "proj": proj,
        "max_proj": max_proj,
        "rgb": rgba[..., :3],
        "curv_quad": curv_quad,
        "mu": w0,
        "H": H_push,
        "bounds": (dlo1, dhi1, dlo2, dhi2, dlob, dhib),
    }


def _ch6_synthetic_quadratic_nll_voxel_cache(bounds, mu, H):
    """Convex quadratic NLL bowl as a voxel lattice (for medium demos)."""
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    H = 0.5 * (H + H.T)
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = tuple(float(v) for v in bounds)
    cell_mult = int(_ch6_nll_voxel_cell_mult())
    gap_pitch = int(_ch6_nll_voxel_gap_pitch())
    n1 = n2 = n3 = int(_g("_ch4_voxel_n_cells")(cell_mult))
    x_edges = _g("_ch4_voxel_cell_edges")(dlo1, dhi1, n1)
    y_edges = _g("_ch4_voxel_cell_edges")(dlo2, dhi2, n2)
    z_edges = _g("_ch4_voxel_cell_edges")(dlob, dhib, n3)
    w1 = 0.5 * (x_edges[:-1] + x_edges[1:])
    w2 = 0.5 * (y_edges[:-1] + y_edges[1:])
    bb = 0.5 * (z_edges[:-1] + z_edges[1:])
    W1, W2, B = np.meshgrid(w1, w2, bb, indexing="ij")
    quad = _ch6_nll_voxel_curv_quad(W1, W2, B, mu, H)
    nll = 0.5 * quad
    g_lo = float(np.nanpercentile(nll, 8.0))
    g_hi = float(np.nanpercentile(nll, 92.0))
    if g_hi <= g_lo + 1e-9:
        g_hi = g_lo + 1.0
    rgba = np.asarray(
        ch4_nll_heatmap_facecolors(nll, vmin=g_lo, vmax=g_hi, alpha=1.0),
        dtype=np.float32,
    )
    curv_quad = quad.astype(np.float32, copy=False)
    if CH6_NLL_VOXEL_USE_CHECKER:
        checker = _ch6_nll_voxel_checker_mask_aniso(
            n1, n2, n3,
            gap_pitch=int(gap_pitch), freq=float(CH6_NLL_VOXEL_CHECKER_FREQ),
        )
    else:
        checker = np.ones((int(n1), int(n2), int(n3)), dtype=bool)
    start = np.array(_g("CH3_LIK_VOXEL_DIAG_START"), dtype=np.float64)
    end = np.array(_g("CH3_LIK_VOXEL_DIAG_END"), dtype=np.float64)
    diag = end - start
    max_proj = float(np.dot(diag, diag))
    proj = (
        (W1 - start[0]) * diag[0]
        + (W2 - start[1]) * diag[1]
        + (B - start[2]) * diag[2]
    ).astype(np.float32, copy=False)
    return {
        "n": (n1, n2, n3),
        "x_edges": x_edges,
        "y_edges": y_edges,
        "z_edges": z_edges,
        "checker": checker,
        "proj": proj,
        "max_proj": max_proj,
        "rgb": rgba[..., :3],
        "curv_quad": curv_quad,
        "mu": mu,
        "H": H,
        "bounds": (dlo1, dhi1, dlo2, dhi2, dlob, dhib),
    }


def _ch6_nll_voxel_walker_alpha_factor_indices(cache, walker_w, i, j, k, *, clear_u):
    """Walker alpha factor for specific voxel indices only (memory-light)."""
    clear_u = float(np.clip(float(clear_u), 0.0, 1.0))
    if clear_u < 1e-6 or walker_w is None or len(i) == 0:
        return None
    w = np.asarray(walker_w, dtype=np.float64).reshape(3)
    x_edges = cache["x_edges"]
    y_edges = cache["y_edges"]
    z_edges = cache["z_edges"]
    w1 = 0.5 * (x_edges[:-1] + x_edges[1:])
    w2 = 0.5 * (y_edges[:-1] + y_edges[1:])
    bb = 0.5 * (z_edges[:-1] + z_edges[1:])
    W1 = w1[np.asarray(i, dtype=np.intp)]
    W2 = w2[np.asarray(j, dtype=np.intp)]
    B = bb[np.asarray(k, dtype=np.intp)]
    H = np.asarray(cache["H"], dtype=np.float64).reshape(3, 3)
    quad = _ch6_nll_voxel_curv_quad(W1, W2, B, w, H)
    ref = float(np.nanpercentile(quad, float(CH6_NLL_VOXEL_WALKER_CLEAR_REF_PCT)))
    ref = max(ref, 1e-9)
    rel = np.clip(quad / ref, 0.0, 4.0)
    boost = np.exp(-rel)
    factor = 1.0 - clear_u * boost
    return np.clip(factor, 0.04, 1.0)


def _ch6_nll_voxel_walker_opaque_factor_indices(cache, walker_w, i, j, k, *, opaque_u):
    """Boost alpha near the walker (medium lights up around the moving point)."""
    opaque_u = float(np.clip(float(opaque_u), 0.0, 1.0))
    if opaque_u < 1e-6 or walker_w is None or len(i) == 0:
        return None
    w = np.asarray(walker_w, dtype=np.float64).reshape(3)
    x_edges = cache["x_edges"]
    y_edges = cache["y_edges"]
    z_edges = cache["z_edges"]
    dx = max(float(x_edges[1] - x_edges[0]), 1e-9)
    dy = max(float(y_edges[1] - y_edges[0]), 1e-9)
    dz = max(float(z_edges[1] - z_edges[0]), 1e-9)
    w1 = 0.5 * (x_edges[:-1] + x_edges[1:])
    w2 = 0.5 * (y_edges[:-1] + y_edges[1:])
    bb = 0.5 * (z_edges[:-1] + z_edges[1:])
    W1 = w1[np.asarray(i, dtype=np.intp)]
    W2 = w2[np.asarray(j, dtype=np.intp)]
    B = bb[np.asarray(k, dtype=np.intp)]
    rel = (
        ((W1 - w[0]) / dx) ** 2
        + ((W2 - w[1]) / dy) ** 2
        + ((B - w[2]) / dz) ** 2
    )
    boost = np.exp(-0.5 * np.clip(rel, 0.0, 12.0))
    factor = 1.0 + opaque_u * boost
    return np.clip(factor, 1.0, 1.0 + opaque_u)


def _ch6_nll_voxel_sweep_diag():
    start = np.array(_g("CH3_LIK_VOXEL_DIAG_START"), dtype=np.float64)
    end = np.array(_g("CH3_LIK_VOXEL_DIAG_END"), dtype=np.float64)
    diag = end - start
    diag_len = float(np.linalg.norm(diag))
    unit = diag / max(diag_len, 1e-9)
    return start, end, diag, diag_len, unit


def _ch6_nll_voxel_clip_visible(cache, visible, clip_bounds):
    if clip_bounds is None:
        return visible
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = (float(v) for v in clip_bounds)
    x_edges = cache["x_edges"]
    y_edges = cache["y_edges"]
    z_edges = cache["z_edges"]
    ii, jj, kk = np.indices(visible.shape)
    cx = 0.5 * (x_edges[ii] + x_edges[ii + 1])
    cy = 0.5 * (y_edges[jj] + y_edges[jj + 1])
    cz = 0.5 * (z_edges[kk] + z_edges[kk + 1])
    in_box = (
        (cx >= dlo1) & (cx <= dhi1)
        & (cy >= dlo2) & (cy <= dhi2)
        & (cz >= dlob) & (cz <= dhib)
    )
    return visible & in_box


def _ch6_nll_voxel_wave_offsets(cache, proj_vals, front, max_proj, wave_amp):
    """Lift voxels along the diagonal sweep direction (peaks at the reveal front)."""
    amp = float(wave_amp)
    if amp <= 1e-6:
        return 0.0, 0.0, 0.0
    _, _, diag, diag_len, unit = _ch6_nll_voxel_sweep_diag()
    band = max(float(max_proj) * 0.16, 1e-9)
    dist_behind = np.clip(float(front) - np.asarray(proj_vals, dtype=np.float64), 0.0, band)
    w = np.sin((dist_behind / band) * np.pi)
    lift = w * max(diag_len * 0.055, 1e-6) * amp
    return lift * unit[0], lift * unit[1], lift * unit[2]


def _ch6_nll_voxel_bar3d(ax3d, cache, mask, *, rgb_override=None, alpha_scale=1.0,
                         uniform_alpha=None, curv_ref_pct=None, wave_amp=0.0, front=0.0,
                         max_proj=1.0):
    if not np.any(mask):
        return
    x_edges = cache["x_edges"]
    y_edges = cache["y_edges"]
    z_edges = cache["z_edges"]
    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    dz = float(z_edges[1] - z_edges[0])
    idx = np.argwhere(mask)
    xs = x_edges[idx[:, 0]].astype(np.float64, copy=True)
    ys = y_edges[idx[:, 1]].astype(np.float64, copy=True)
    zs = z_edges[idx[:, 2]].astype(np.float64, copy=True)
    if float(wave_amp) > 1e-6:
        ox, oy, oz = _ch6_nll_voxel_wave_offsets(
            cache, cache["proj"][mask], front, max_proj, wave_amp,
        )
        xs = xs + ox
        ys = ys + oy
        zs = zs + oz
    if curv_ref_pct is None:
        curv_ref_pct = float(CH6_NLL_VOXEL_CURV_REF_PCT)
    alpha = _ch6_nll_voxel_alpha_from_quad(cache["curv_quad"], curv_ref_pct)[mask]
    if cache.get("obj_alpha") is not None:
        alpha = np.asarray(cache["obj_alpha"], dtype=np.float64)[mask]
    elif uniform_alpha is not None:
        alpha = np.full(int(np.count_nonzero(mask)), float(uniform_alpha), dtype=np.float64)
    alpha = alpha * float(np.clip(float(alpha_scale), 0.0, 1.0))
    if rgb_override is not None:
        rgb_vis = np.clip(np.asarray(rgb_override, dtype=np.float64), 0.0, 1.0)
    else:
        rgb_vis = np.clip(np.asarray(cache["rgb"], dtype=np.float64)[mask], 0.0, 1.0)
    colors = np.column_stack([rgb_vis, np.clip(alpha, 0.0, 1.0)])
    ax3d.bar3d(
        xs, ys, zs, dx, dy, dz,
        color=colors,
        shade=False,
        linewidth=0.0,
        edgecolor=(0.0, 0.0, 0.0, 0.0),
        zorder=10,
    )


def _ch6_draw_nll_voxels(
    ax3d,
    cache,
    sweep_u,
    *,
    curv_ref_pct=None,
    alpha_scale=1.0,
    uniform_alpha=None,
    walker_w=None,
    walker_clear_u=0.0,
    walker_opaque_u=0.0,
    prev_cache=None,
    color_blend_u=1.0,
    clip_bounds=None,
    sweep_wave_amp=0.0,
    sweep_reveal_blend=False,
):
    """Diagonal corner sweep (Ch4-style) with per-voxel NLL transparency."""
    if curv_ref_pct is None:
        curv_ref_pct = float(CH6_NLL_VOXEL_CURV_REF_PCT)
    u = float(np.clip(float(sweep_u), 0.0, 1.0))
    u = float(_g("ch3_knob_smoothstep")(u))
    front = u * float(cache["max_proj"])
    max_proj = float(cache["max_proj"])
    checker = cache["checker"]
    proj = cache["proj"]
    wave_amp = float(sweep_wave_amp)
    reveal = bool(sweep_reveal_blend) and prev_cache is not None and u < 1.0 - 1e-9

    if reveal:
        swept = _ch6_nll_voxel_clip_visible(
            cache, (proj <= front + 1e-9) & checker, clip_bounds,
        )
        pending = _ch6_nll_voxel_clip_visible(
            cache, (proj > front + 1e-9) & checker, clip_bounds,
        )
        if np.any(pending):
            _ch6_nll_voxel_bar3d(
                ax3d, cache, pending,
                rgb_override=np.asarray(prev_cache["rgb"], dtype=np.float64)[pending],
                alpha_scale=alpha_scale,
                uniform_alpha=uniform_alpha,
                curv_ref_pct=curv_ref_pct,
            )
        if np.any(swept):
            prev_rgb = np.asarray(prev_cache["rgb"], dtype=np.float64)[swept]
            new_rgb = np.asarray(cache["rgb"], dtype=np.float64)[swept]
            band = max(max_proj * 0.14, 1e-9)
            dist_behind = np.clip(front - proj[swept], 0.0, band)
            local_u = np.sin((dist_behind / band) * 0.5 * np.pi).reshape(-1, 1)
            rgb_blend = (1.0 - local_u) * prev_rgb + local_u * new_rgb
            _ch6_nll_voxel_bar3d(
                ax3d, cache, swept,
                rgb_override=rgb_blend,
                alpha_scale=alpha_scale,
                uniform_alpha=uniform_alpha,
                curv_ref_pct=curv_ref_pct,
                wave_amp=wave_amp,
                front=front,
                max_proj=max_proj,
            )
        return

    visible = _ch6_nll_voxel_clip_visible(cache, (proj <= front + 1e-9) & checker, clip_bounds)
    if not np.any(visible):
        return
    x_edges = cache["x_edges"]
    y_edges = cache["y_edges"]
    z_edges = cache["z_edges"]
    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    dz = float(z_edges[1] - z_edges[0])
    idx = np.argwhere(visible)
    xs = x_edges[idx[:, 0]].astype(np.float64, copy=True)
    ys = y_edges[idx[:, 1]].astype(np.float64, copy=True)
    zs = z_edges[idx[:, 2]].astype(np.float64, copy=True)
    if wave_amp > 1e-6:
        ox, oy, oz = _ch6_nll_voxel_wave_offsets(
            cache, proj[visible], front, max_proj, wave_amp,
        )
        xs = xs + ox
        ys = ys + oy
        zs = zs + oz
    alpha = _ch6_nll_voxel_alpha_from_quad(cache["curv_quad"], curv_ref_pct)[visible]
    if cache.get("obj_alpha") is not None:
        alpha = np.asarray(cache["obj_alpha"], dtype=np.float64)[visible]
    elif uniform_alpha is not None:
        u_alpha = float(uniform_alpha)
        alpha = np.full(int(np.count_nonzero(visible)), u_alpha, dtype=np.float64)
    alpha = alpha * float(np.clip(float(alpha_scale), 0.0, 1.0))
    if float(walker_clear_u) > 1e-6 and walker_w is not None:
        wf = _ch6_nll_voxel_walker_alpha_factor_indices(
            cache, walker_w, idx[:, 0], idx[:, 1], idx[:, 2],
            clear_u=walker_clear_u,
        )
        if wf is not None:
            alpha = alpha * wf
    if float(walker_opaque_u) > 1e-6 and walker_w is not None:
        wf = _ch6_nll_voxel_walker_opaque_factor_indices(
            cache, walker_w, idx[:, 0], idx[:, 1], idx[:, 2],
            opaque_u=walker_opaque_u,
        )
        if wf is not None and uniform_alpha is not None and cache.get("obj_alpha") is None:
            wake = float(CH6_VOXEL_MEDIUM_WAKE_ALPHA)
            blend = np.clip((wf - 1.0) / max(float(walker_opaque_u), 1e-6), 0.0, 1.0)
            alpha = u_alpha + (wake - u_alpha) * blend
        elif wf is not None:
            alpha = np.clip(alpha * wf, 0.0, 1.0)
    blend_u = float(np.clip(float(color_blend_u), 0.0, 1.0))
    rgb_vis = np.asarray(cache["rgb"], dtype=np.float64)[visible]
    if prev_cache is not None and blend_u < 1.0 - 1e-9:
        prev_rgb = np.asarray(prev_cache["rgb"], dtype=np.float64)[visible]
        rgb_vis = (1.0 - blend_u) * prev_rgb + blend_u * rgb_vis
    colors = np.column_stack([rgb_vis, alpha])
    ax3d.bar3d(
        xs, ys, zs, dx, dy, dz,
        color=colors,
        shade=False,
        linewidth=0.0,
        edgecolor=(0.0, 0.0, 0.0, 0.0),
        zorder=10,
    )


def _ch6_nll_voxel_trail_subdiv():
    from ch6_frequentist import _CH3_DRAFT

    return int(
        CH6_NLL_VOXEL_TRAIL_SUBDIV_DRAFT if _CH3_DRAFT else CH6_NLL_VOXEL_TRAIL_SUBDIV_HQ
    )


def _ch6_nll_voxel_trail_half_cells():
    from ch6_frequentist import _CH3_DRAFT

    return int(
        CH6_NLL_VOXEL_TRAIL_HALF_CELLS_DRAFT
        if _CH3_DRAFT
        else CH6_NLL_VOXEL_TRAIL_HALF_CELLS_HQ
    )


def _ch6_voxel_cache_lookup_rgba(w1, w2, wb, cache, *, alpha=0.94):
    """Sample ch6_127 landscape RGBA at nearest voxel-grid cell centers."""
    x_edges = np.asarray(cache["x_edges"], dtype=np.float64)
    y_edges = np.asarray(cache["y_edges"], dtype=np.float64)
    z_edges = np.asarray(cache["z_edges"], dtype=np.float64)
    n_raw = cache["n"]
    if isinstance(n_raw, (tuple, list)):
        n1, n2, n3 = (int(v) for v in n_raw)
    else:
        n1 = n2 = n3 = int(n_raw)
    w1 = np.asarray(w1, dtype=np.float64).reshape(-1)
    w2 = np.asarray(w2, dtype=np.float64).reshape(-1)
    wb = np.asarray(wb, dtype=np.float64).reshape(-1)

    def _idx(w, edges, n):
        return np.clip(np.searchsorted(edges, w, side="right") - 1, 0, n - 1)

    ii = _idx(w1, x_edges, n1)
    jj = _idx(w2, y_edges, n2)
    kk = _idx(wb, z_edges, n3)
    rgb = np.asarray(cache["rgb"], dtype=np.float64)[ii, jj, kk]
    if float(rgb.max()) > 1.0 + 1e-6:
        rgb = rgb / 255.0
    if rgb.ndim == 1:
        rgb = rgb.reshape(1, -1)
    out = np.ones((len(w1), 4), dtype=np.float64)
    out[:, :3] = rgb[:, :3] if rgb.shape[1] >= 3 else rgb
    out[:, 3] = float(alpha)
    return out


def _ch6_nll_voxel_trail_checker_mask(shape, *, freq=None):
    """Sparse lattice inside trail clusters (same freq tiers as landscape voxels)."""
    n1, n2, n3 = (int(v) for v in shape)
    freq = float(CH6_NLL_VOXEL_TRAIL_CHECKER_FREQ if freq is None else freq)
    freq = float(np.clip(freq, 0.0, 1.0))
    if freq >= 1.0 - 1e-9:
        return np.ones((n1, n2, n3), dtype=bool)
    ii, jj, kk = np.indices((n1, n2, n3))
    stride = 2 if freq >= 0.25 - 1e-9 else 3
    return (ii % stride == 0) & (jj % stride == 0) & (kk % stride == 0)


def _ch6_nll_voxel_trail_cluster(
    center,
    study,
    exam,
    y,
    *,
    cell_w12,
    cell_b,
    half_cells_w12,
    half_cells_b,
    vmin,
    vmax,
    alpha=0.94,
):
    """Anisotropic voxel cluster; per-cell NLL colors on the ch6_127 global scale."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    hc12 = int(half_cells_w12)
    hcb = int(half_cells_b)
    dx12 = float(cell_w12)
    dzb = float(cell_b)
    if dx12 <= 1e-12 or dzb <= 1e-12 or hc12 < 0 or hcb < 0:
        return None
    g12 = np.arange(-hc12, hc12 + 1, dtype=np.float64)
    gb = np.arange(-hcb, hcb + 1, dtype=np.float64)
    g1, g2, g3 = np.meshgrid(g12, g12, gb, indexing="ij")
    sparse = _ch6_nll_voxel_trail_checker_mask(g1.shape)
    if not np.any(sparse):
        return None
    w1 = (center[0] + g1[sparse] * dx12).ravel()
    w2 = (center[1] + g2[sparse] * dx12).ravel()
    wb = (center[2] + g3[sparse] * dzb).ravel()
    nll = _g("_ch3_nll_sum_on_flat_grid")(study, exam, y, w1, w2, wb)
    colors = np.asarray(
        ch4_nll_heatmap_facecolors(
            nll, vmin=float(vmin), vmax=float(vmax), alpha=float(alpha),
        ),
        dtype=np.float64,
    )
    if colors.ndim == 1:
        colors = colors.reshape(1, -1)
    return {
        "xs": w1 - 0.5 * dx12,
        "ys": w2 - 0.5 * dx12,
        "zs": wb - 0.5 * dzb,
        "dx": dx12,
        "dy": dx12,
        "dz": dzb,
        "colors": colors,
        "cell_key": (
            int(np.round(center[0] / dx12)),
            int(np.round(center[1] / dx12)),
            int(np.round(center[2] / dzb)),
        ),
    }


def _ch6_nll_voxel_trail_params(w12_span, b_span):
    """Cell sizes and half-widths for a screen-cubic cluster with full b-axis view."""
    w12_span = max(float(w12_span), 1e-6)
    b_span = max(float(b_span), 1e-6)
    b_ratio = b_span / w12_span
    hc_w12 = int(_ch6_nll_voxel_trail_half_cells())
    hc_b = min(max(hc_w12, int(round(hc_w12 * b_ratio))), hc_w12 * 12)
    cell_w12 = w12_span / float(_ch6_nll_voxel_trail_subdiv())
    cell_b = cell_w12
    return cell_w12, cell_b, hc_w12, hc_b


def _ch6_nll_voxel_trail_global_cache(study, exam, y, *, bounds):
    """Fine classroom-NLL voxel grid on ``bounds`` (ch4 ball-voxel style)."""
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = (float(v) for v in bounds)
    n = int(_ch6_nll_voxel_trail_subdiv())
    x_edges = _g("_ch4_voxel_cell_edges")(dlo1, dhi1, n)
    y_edges = _g("_ch4_voxel_cell_edges")(dlo2, dhi2, n)
    z_edges = _g("_ch4_voxel_cell_edges")(dlob, dhib, n)
    w1 = 0.5 * (x_edges[:-1] + x_edges[1:])
    w2 = 0.5 * (y_edges[:-1] + y_edges[1:])
    bb = 0.5 * (z_edges[:-1] + z_edges[1:])
    W1, W2, B = np.meshgrid(w1, w2, bb, indexing="ij")
    nll = _g("_ch3_nll_sum_on_flat_grid")(
        study, exam, y,
        W1.ravel(), W2.ravel(), B.ravel(),
    ).reshape(W1.shape)
    g_lo, g_hi = _g("ch4_nll_global_scale")()
    rgba = np.asarray(
        ch4_nll_heatmap_facecolors(nll, vmin=g_lo, vmax=g_hi, alpha=0.94),
        dtype=np.float64,
    )
    checker = _g("_ch4_voxel_checker_mask")(n, gap_pitch=1)
    return {
        "n": n,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "z_edges": z_edges,
        "checker": checker,
        "rgba": rgba,
        "dx": float(x_edges[1] - x_edges[0]),
        "dy": float(y_edges[1] - y_edges[0]),
        "dz": float(z_edges[1] - z_edges[0]),
    }


def _ch6_ball_voxel_half_from_bounds(bounds, *, cell_mult=None):
    """Half-extent of the local voxel cluster in data units."""
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = (float(v) for v in bounds)
    n = float(_ch6_nll_voxel_trail_subdiv())
    dx = min(
        (dhi1 - dlo1) / n,
        (dhi2 - dlo2) / n,
        (dhib - dlob) / n,
    )
    mult = float(CH6_NLL_VOXEL_TRAIL_HALF_CELL_MULT if cell_mult is None else cell_mult)
    return max(mult * dx, dx * 1.5)


def _ch6_ball_voxel_build_cube(global_cache, center, half):
    """Ball-sized checkerboard subset around ``center`` (matches ch4 Newton voxel cubes)."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    half = float(half)
    x_edges = global_cache["x_edges"]
    y_edges = global_cache["y_edges"]
    z_edges = global_cache["z_edges"]
    w1 = 0.5 * (x_edges[:-1] + x_edges[1:])
    w2 = 0.5 * (y_edges[:-1] + y_edges[1:])
    bb = 0.5 * (z_edges[:-1] + z_edges[1:])
    W1, W2, B = np.meshgrid(w1, w2, bb, indexing="ij")
    in_ball = (
        (np.abs(W1 - center[0]) <= half)
        & (np.abs(W2 - center[1]) <= half)
        & (np.abs(B - center[2]) <= half)
    )
    checker = np.asarray(global_cache["checker"], dtype=bool) & in_ball
    ci = int(np.argmin(np.abs(w1 - center[0])))
    cj = int(np.argmin(np.abs(w2 - center[1])))
    ck = int(np.argmin(np.abs(bb - center[2])))
    n = int(global_cache["n"])
    ii, jj, kk = np.indices((n, n, n))
    cheb = np.maximum(np.maximum(np.abs(ii - ci), np.abs(jj - cj)), np.abs(kk - ck))
    max_cheb = int(cheb[checker].max()) if np.any(checker) else 0
    return {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "z_edges": z_edges,
        "checker": checker,
        "cheb": cheb,
        "max_cheb": max_cheb,
        "rgba": global_cache["rgba"],
        "dx": float(global_cache["dx"]),
        "dy": float(global_cache["dy"]),
        "dz": float(global_cache["dz"]),
        "ci": ci,
        "cj": cj,
        "ck": ck,
    }


def _ch6_ball_voxel_cube_draw(cube, *, grow_u=1.0, alpha_scale=1.0):
    grow_u = float(np.clip(float(grow_u), 0.0, 1.0))
    max_cheb = float(cube["max_cheb"])
    lim = grow_u * max_cheb + 1e-9
    visible = cube["checker"] & (cube["cheb"] <= lim)
    if not np.any(visible):
        return None
    idx = np.argwhere(visible)
    xs = cube["x_edges"][idx[:, 0]]
    ys = cube["y_edges"][idx[:, 1]]
    zs = cube["z_edges"][idx[:, 2]]
    colors = cube["rgba"][visible].copy()
    if float(alpha_scale) < 1.0 - 1e-6:
        colors[..., 3] *= float(alpha_scale)
    return {
        "xs": xs,
        "ys": ys,
        "zs": zs,
        "dx": cube["dx"],
        "dy": cube["dy"],
        "dz": cube["dz"],
        "colors": colors,
    }


def _ch6_ball_voxel_merge_draws(draws):
    draws = [d for d in draws if d is not None]
    if not draws:
        return None
    if len(draws) == 1:
        return draws[0]
    return {
        "xs": np.concatenate([d["xs"] for d in draws]),
        "ys": np.concatenate([d["ys"] for d in draws]),
        "zs": np.concatenate([d["zs"] for d in draws]),
        "dx": draws[0]["dx"],
        "dy": draws[0]["dy"],
        "dz": draws[0]["dz"],
        "colors": np.concatenate([d["colors"] for d in draws], axis=0),
    }


def _ch6_draw_nll_voxel_bar3d(ax3d, draw):
    """Draw a bar3d voxel batch (single stamp or merged trail)."""
    if draw is None:
        return
    ax3d.bar3d(
        draw["xs"], draw["ys"], draw["zs"],
        draw["dx"], draw["dy"], draw["dz"],
        color=draw["colors"],
        shade=False,
        linewidth=0.0,
        edgecolor=(0.0, 0.0, 0.0, 0.0),
        zorder=1,
    )


def _ch6_draw_nll_voxel_trail(ax3d, trail):
    """Draw merged voxel trail (ch4 Newton ball-voxel clusters)."""
    if trail is None:
        return
    if isinstance(trail, dict) and "xs" in trail:
        _ch6_draw_nll_voxel_bar3d(ax3d, trail)
        return
    for stamp in trail:
        _ch6_draw_nll_voxel_bar3d(ax3d, stamp)


def _ch6_draw_nll_ct_slice(
    ax3d,
    study,
    exam,
    y,
    bounds,
    *,
    sweep_axis=None,
    plane_val=None,
    pivot_from=None,
    pivot_to=None,
    pivot_u=0.0,
    alpha=1.0,
):
    """Axis-aligned or pivot NLL heatmap slice (Ch4 CT scan geometry)."""
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    al = float(np.clip(float(alpha), 0.0, 1.0))
    if pivot_from is not None and pivot_to is not None:
        w1, w2, b, nll = _g("_ch4_ct_pivot_mesh")(
            study, exam, y,
            str(pivot_from), str(pivot_to), bounds,
            theta_u=float(pivot_u),
        )
    else:
        w1, w2, b = _g("_ch4_ct_sweep_mesh")(str(sweep_axis), float(plane_val), bounds)
        nll = _g("_ch4_ct_nll_at_grid")(study, exam, y, w1, w2, b)
    g_lo, g_hi = _g("ch4_nll_global_scale")()
    face = ch4_nll_heatmap_facecolors(nll, vmin=g_lo, vmax=g_hi, alpha=al)
    ax3d.plot_surface(
        w1, w2, b,
        facecolors=face,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        zorder=5,
    )


def _ch6_style_nll_voxel_ax3d(ax3d, bounds, *, elev, azim, camera_zoom=1.0):
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = bounds
    _g("_ch3_lik_style_ax3d")(ax3d, dlo1, dhi1, dlo2, dhi2, dlob, dhib)
    ax3d.tick_params(labelsize=6)
    _g("ch4_lik_ct_view_init")(ax3d, cam_azim_u=0.0, cam_spin_deg=0.0)
    ax3d.view_init(elev=float(elev), azim=float(azim))
    ax3d.computed_zorder = False
    zoom = float(max(camera_zoom, 1e-3))
    if hasattr(ax3d, "dist"):
        ax3d.dist = float(np.clip(10.0 / zoom, 3.0, 14.0))


def _variance_init_view():
    base_elev = float(ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV"))))
    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))
    return base_elev, base_azim


def _variance_resolve_view(name, *, flip_180: bool = False):
    if name == "init":
        elev, azim = _variance_init_view()
    else:
        elev, azim = CH6_VARIANCE_VIEWS[name]
        elev, azim = float(elev), float(azim)
    if flip_180 and name in ("w_st_face", "w_el_face"):
        azim = (float(azim) + 180.0) % 360.0
    return float(elev), float(azim)


def _variance_interp_view(name_a, name_b, u, *, flip_b: bool = False):
    ea, aa = _variance_resolve_view(name_a)
    eb, ab = _variance_resolve_view(name_b, flip_180=bool(flip_b))
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


def _variance_belief_marker_colors(W, *, revealed, grey_u, z_lim):
    """Belief-density marker colors for variance on a zoomed cloud."""
    W = np.asarray(W, dtype=np.float64)
    n = len(W)
    full = _ch6_density_belief_rgba(_ch6_kde_density_values(W), z_lim=z_lim, alpha=0.92)
    grey = np.array(plt.matplotlib.colors.to_rgb(CH6_VARIANCE_GREY), dtype=np.float64)
    cols = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        rgba = np.asarray(full[i], dtype=np.float64)
        if bool(revealed[i]):
            cols[i] = rgba
        else:
            u = float(np.clip(grey_u, 0.0, 1.0))
            rgb = rgba[:3] * (1.0 - u) + grey * u
            al = float(rgba[3]) * (0.35 + 0.65 * (1.0 - u))
            cols[i] = (*rgb.tolist(), al)
    return cols


def _variance_belief_marker_colors_w12(W, *, revealed, grey_u, z_lim):
    """Belief-density colors from a 2D KDE in the w_ST–w_EL plane (projected clouds)."""
    W = np.asarray(W, dtype=np.float64)
    n = len(W)
    w12 = W[:, :2]
    if n >= 5:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(w12.T)
        d = np.maximum(kde(w12.T), 1e-12)
    else:
        d = np.ones(n, dtype=np.float64)
    full = _ch6_density_belief_rgba(d, z_lim=z_lim, alpha=0.92)
    grey = np.array(plt.matplotlib.colors.to_rgb(CH6_VARIANCE_GREY), dtype=np.float64)
    cols = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        rgba = np.asarray(full[i], dtype=np.float64)
        if bool(revealed[i]):
            cols[i] = rgba
        else:
            u = float(np.clip(grey_u, 0.0, 1.0))
            rgb = rgba[:3] * (1.0 - u) + grey * u
            al = float(rgba[3]) * (0.35 + 0.65 * (1.0 - u))
            cols[i] = (*rgb.tolist(), al)
    return cols


def _ch6_cov_zoom_lim_stpos_elneg(W, mu, axis_lim, *, pad_frac=0.10):
    """Zoom to w_ST above μ and w_EL below μ (positive-ST / negative-EL quadrant)."""
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    lo0, hi0 = float(axis_lim[0]), float(axis_lim[1])
    st_lo = float(mu[0])
    st_hi = float(max(W[:, 0].max(), mu[0] + 1e-4))
    el_lo = float(min(W[:, 1].min(), mu[1] - 1e-4))
    el_hi = float(mu[1])
    pad_st = float(pad_frac) * max(st_hi - st_lo, 1e-6)
    pad_el = float(pad_frac) * max(el_hi - el_lo, 1e-6)
    lo = max(lo0, min(el_lo - pad_el, st_lo - pad_st * 0.2))
    hi = min(hi0, max(st_hi + pad_st, el_hi + pad_el * 0.2))
    mid = 0.5 * (lo + hi)
    half = 0.5 * max(hi - lo, (st_hi - st_lo) + pad_st, (el_hi - el_lo) + pad_el)
    lo = max(lo0, mid - half)
    hi = min(hi0, mid + half)
    return (lo, hi)


def _ch6_cov_draw_view_axis_signs(
    ax3d, mu, w_st_lim, w_el_lim, *, reveal_u=1.0, fontsize=50,
):
    """Place +/− on each axis half within the visible w_ST / w_EL windows."""
    u = float(np.clip(reveal_u, 0.0, 1.0))
    if u <= 1e-6:
        return
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    b = float(mu[2])
    st_lo, st_hi = float(w_st_lim[0]), float(w_st_lim[1])
    el_lo, el_hi = float(w_el_lim[0]), float(w_el_lim[1])
    pc, nc = CH6_VARIANCE_POS_COLOR, CH6_VARIANCE_NEG_COLOR
    st_p = mu.copy(); st_p[0] = 0.5 * (max(mu[0], st_lo) + st_hi)
    st_m = mu.copy(); st_m[0] = 0.5 * (st_lo + min(mu[0], st_hi))
    el_p = mu.copy(); el_p[1] = 0.5 * (max(mu[1], el_lo) + el_hi)
    el_m = mu.copy(); el_m[1] = 0.5 * (el_lo + min(mu[1], el_hi))
    for pos, char, col in (
        (st_p, "+", pc), (st_m, "−", nc),
        (el_p, "+", pc), (el_m, "−", nc),
    ):
        ax3d.text(
            float(pos[0]), float(pos[1]), b, char,
            color=col, fontsize=float(fontsize), fontweight="bold",
            ha="center", va="center", alpha=0.96 * u, zorder=18,
        )


def _ch6_cov_draw_quadrant_products(
    ax3d, mu, w_st_lim, w_el_lim, quads, *, reveal_u=1.0, fontsize=36,
):
    """Factor signs and product sign per quadrant; ``=`` in black."""
    u = float(np.clip(reveal_u, 0.0, 1.0))
    if u <= 1e-6 or not quads:
        return
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    st_lo, st_hi = float(w_st_lim[0]), float(w_st_lim[1])
    el_lo, el_hi = float(w_el_lim[0]), float(w_el_lim[1])
    b = float(mu[2])
    st_span = max(st_hi - st_lo, 1e-6)
    el_span = max(el_hi - el_lo, 1e-6)
    dx = 0.075 * st_span
    x_left = -0.14 * st_span
    y_up = 0.11 * el_span
    pc = CH6_VARIANCE_POS_COLOR
    nc = CH6_VARIANCE_NEG_COLOR
    pur = CH6_VARIANCE_NEG_COLOR
    pos = CH6_COV_PRODUCT_POS_COLOR
    eq = CH6_COV_EQ_COLOR
    specs = {
        "pp": (
            0.5 * (max(mu[0], st_lo) + st_hi) + x_left,
            0.5 * (max(mu[1], el_lo) + el_hi) + y_up,
            [("+", pc), ("+", pc), ("=", eq), ("+", pos)],
        ),
        "nn": (
            0.5 * (st_lo + min(mu[0], st_hi)) + x_left,
            0.5 * (el_lo + min(mu[1], el_hi)),
            [("−", nc), ("−", nc), ("=", eq), ("+", pos)],
        ),
        "pn": (
            0.5 * (max(mu[0], st_lo) + st_hi) + x_left,
            0.5 * (el_lo + min(mu[1], el_hi)) + y_up,
            [("+", pc), ("−", nc), ("=", eq), ("−", pur)],
        ),
        "np": (
            0.5 * (st_lo + min(mu[0], st_hi)) + x_left,
            0.5 * (max(mu[1], el_lo) + el_hi),
            [("−", nc), ("+", pc), ("=", eq), ("−", pur)],
        ),
    }
    for q in quads:
        if q not in specs:
            continue
        x0, y0, tokens = specs[q]
        for k, (char, col) in enumerate(tokens):
            ax3d.text(
                float(x0 + k * dx), float(y0), b, char,
                color=col, fontsize=float(fontsize), fontweight="bold",
                ha="center", va="center", alpha=0.96 * u, zorder=20,
            )


def _ch6_cov_marker_override(n_pts, mu, W, indices, *, mode="mismatch"):
    """Per-point RGBA overrides for highlighted covariance demo points."""
    from matplotlib.colors import to_rgba

    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    W = np.asarray(W, dtype=np.float64)
    out = np.zeros((int(n_pts), 4), dtype=np.float64)
    for i in indices:
        ii = int(i)
        dst = W[ii] - mu
        if mode == "mismatch":
            out[ii] = to_rgba(CH6_COV_MISMATCH_COLOR, 0.97)
        elif mode == "product":
            out[ii] = to_rgba("#111111", 0.98)
        elif mode == "matched":
            if dst[0] > 0.0 and dst[1] > 0.0:
                out[ii] = to_rgba(CH6_VARIANCE_POS_COLOR, 0.98)
            elif dst[0] < 0.0 and dst[1] < 0.0:
                out[ii] = to_rgba(CH6_VARIANCE_NEG_COLOR, 0.98)
            elif dst[0] > 0.0:
                out[ii] = to_rgba(CH6_VARIANCE_POS_COLOR, 0.98)
            else:
                out[ii] = to_rgba(CH6_VARIANCE_NEG_COLOR, 0.98)
    return out


def _ch6_cov_view_axis_lim(W, mu, axis_lim, *, pad_frac=0.08):
    """Tighter square limits framing the w_ST–w_EL cloud around μ."""
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    lo0, hi0 = float(axis_lim[0]), float(axis_lim[1])
    lo_d = float(min(W[:, 0].min(), W[:, 1].min(), mu[0], mu[1]))
    hi_d = float(max(W[:, 0].max(), W[:, 1].max(), mu[0], mu[1]))
    span = max(hi_d - lo_d, 1e-6)
    pad = float(pad_frac) * span
    lo = max(lo0, lo_d - pad)
    hi = min(hi0, hi_d + pad)
    if hi - lo < 0.18 * (hi0 - lo0):
        mid = 0.5 * (lo_d + hi_d)
        half = 0.5 * max(span + 2.0 * pad, 0.18 * (hi0 - lo0))
        lo = max(lo0, mid - half)
        hi = min(hi0, mid + half)
    return (lo, hi)


def _variance_sign_color(mu, pt, axis, *, pos_color, neg_color, mix_color):
    d = float(np.asarray(pt, dtype=np.float64).reshape(3)[int(axis)]) - float(
        np.asarray(mu, dtype=np.float64).reshape(3)[int(axis)]
    )
    if abs(d) < 1e-8:
        return mix_color
    return pos_color if d > 0.0 else neg_color


def _variance_plot_seg(ax3d, p0, p1, *, color, lw, ls, alpha, zorder=7):
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    if float(alpha) <= 1e-4:
        return
    ax3d.plot(
        [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
        color=color, lw=float(lw), ls=ls, alpha=float(alpha), zorder=float(zorder),
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


def _variance_draw_axis_component(
    ax3d, mu, pt, axis, *, alpha=1.0, color=CH6_VARIANCE_RED,
    solid_lw=2.2, dotted_lw=1.8,
):
    corner = _variance_axis_corner(mu, pt, axis)
    dotted_end = _variance_dotted_end(mu, pt, axis)
    _variance_plot_seg(
        ax3d, mu, corner, color=color, lw=float(solid_lw), ls="-", alpha=alpha,
    )
    _variance_plot_seg(
        ax3d, pt, dotted_end, color=color, lw=float(dotted_lw), ls=":", alpha=alpha,
    )


def _variance_draw_axis_sign_labels(
    ax3d,
    mu,
    axis,
    axis_lim,
    *,
    pos_color=CH6_VARIANCE_POS_COLOR,
    neg_color=CH6_VARIANCE_NEG_COLOR,
    reveal_u=1.0,
    fontsize=42,
):
    u = float(np.clip(reveal_u, 0.0, 1.0))
    if u <= 1e-6:
        return
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    plus = mu.copy()
    minus = mu.copy()
    plus[int(axis)] = hi
    minus[int(axis)] = lo
    ax3d.text(
        float(plus[0]), float(plus[1]), float(plus[2]), "+",
        color=pos_color, fontsize=float(fontsize), fontweight="bold",
        ha="center", va="center", alpha=0.95 * u, zorder=15,
    )
    ax3d.text(
        float(minus[0]), float(minus[1]), float(minus[2]), "−",
        color=neg_color, fontsize=float(fontsize), fontweight="bold",
        ha="center", va="center", alpha=0.95 * u, zorder=15,
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
    color=CH6_VARIANCE_RED,
    signed_halves=False,
    pos_color=None,
    neg_color=None,
):
    """Axis-aligned segment through the cloud mean."""
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    a = mu.copy()
    b = mu.copy()
    a[int(axis)] = float(lo)
    b[int(axis)] = float(hi)
    if signed_halves:
        pc = pos_color or CH6_VARIANCE_POS_COLOR
        nc = neg_color or CH6_VARIANCE_NEG_COLOR
        mid = mu.copy()
        _variance_plot_seg(ax3d, a, mid, color=nc, lw=lw, ls="-", alpha=alpha)
        _variance_plot_seg(ax3d, mid, b, color=pc, lw=lw, ls="-", alpha=alpha)
    if labels and float(alpha) > 0.35:
        lbl = r"$\sigma^2$"
        ax3d.text(
            float(a[0]), float(a[1]), float(a[2]), lbl,
                color=nc, fontsize=9, ha="right", va="center",
        )
        ax3d.text(
            float(b[0]), float(b[1]), float(b[2]), lbl,
                color=pc, fontsize=9, ha="left", va="center",
            )
        return
    _variance_plot_seg(ax3d, a, b, color=color, lw=lw, ls="-", alpha=alpha)
    if labels and float(alpha) > 0.35:
        lbl = r"$\sigma^2$"
        ax3d.text(
            float(a[0]), float(a[1]), float(a[2]), lbl,
            color=color, fontsize=9, ha="right", va="center",
        )
        ax3d.text(
            float(b[0]), float(b[1]), float(b[2]), lbl,
            color=color, fontsize=9, ha="left", va="center",
        )


def _variance_draw_axis_through(
    ax3d, mu, axis, lo, hi, *, alpha=1.0, lw=2.4, color=CH6_VARIANCE_RED,
    signed_halves=False, pos_color=None, neg_color=None,
):
    _variance_draw_centered_range(
        ax3d, mu, axis, lo, hi, alpha=alpha, labels=False, lw=lw, color=color,
        signed_halves=signed_halves, pos_color=pos_color, neg_color=neg_color,
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


def _variance_draw_box(
    ax3d, bounds, *, edge_alpha=0.85, fill_alpha=0.06, inner_bounds=None, gap_u=0.0,
    color=CH6_VARIANCE_RED,
):
    corners, edges = _variance_box_edges(bounds)
    for i, j in edges:
        _variance_plot_seg(
            ax3d, corners[i], corners[j],
            color=color, lw=1.8, ls="-", alpha=edge_alpha,
        )
    if inner_bounds is not None and float(gap_u) > 1e-4:
        ic, ie = _variance_box_edges(inner_bounds)
        for i, j in ie:
            _variance_plot_seg(
                ax3d, ic[i], ic[j],
                color=color, lw=1.4, ls=":", alpha=0.55 * gap_u,
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
                color=color, alpha=float(fill_alpha) * float(gap_u),
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
    camera_zoom=1.0,
    marker_s=28,
    knob_w=None,
    ghost_ws=None,
    ghost_fade_u=1.0,
    ghost_density_reveal_u=1.0,
    belief_z_lim=None,
    mu_threshold_2d=None,
    mu_threshold_2d_u=0.0,
    mu_threshold_2d_lw=3.4,
    mu_threshold_2d_color=CH6_VARIANCE_RED,
    component_color=None,
    range_color=None,
    signed_components=False,
    signed_range_halves=False,
    sign_labels=False,
    sign_labels_u=1.0,
    box_color=CH6_VARIANCE_RED,
    marker_alpha_scale=1.0,
    hessian_arrow_dirs=None,
    hessian_arrow_lengths=None,
    hessian_arrow_reveal_u=0.0,
    hessian_arrow_color=CH6_HESSIAN_ARROW_COLOR,
    hessian_arrow_lw=3.0,
    hessian_arrow_axis_lim=None,
    hessian_arrow_w=None,
    cov_arrow_dirs=None,
    cov_arrow_lengths=None,
    cov_arrow_reveal_u=0.0,
    box_fade_u=1.0,
    plane_b_value=None,
    cov_accum_length=0.0,
    cov_accum_u=0.0,
    cov_accum_color=CH6_COV_COLOR,
    sign_label_fontsize=42,
    view_axis_lim=None,
    view_w_st_lim=None,
    view_w_el_lim=None,
    belief_w12=False,
    marker_bright_mask=None,
    marker_dim_scale=0.22,
    marker_color_override=None,
    marker_s_scale=1.0,
    marker_s_per_point=None,
    marker_below_voxels=False,
    axis_component_solid_lw=2.2,
    axis_component_dotted_lw=1.8,
    cov_view_signs=False,
    sign_label_view_lim=None,
    cov_quadrant_products=None,
    cov_quadrant_products_u=0.0,
    cov_left_fade_u=0.0,
    cov_math_header_u=0.0,
    cov_contrib_landed=None,
    cov_contrib_fly=None,
    cov_contrib_ymax=None,
    cov_beat_caption=None,
    cov_beat_caption_u=0.0,
    cov_math_header_block=None,
    cov_positive_only_pile=False,
    cov_var_axis=0,
    cov_var_flying=False,
    matrix_C=None,
    matrix_var_u=None,
    matrix_cov_u=None,
    matrix_cov_visible_cols=None,
    matrix_arrow_boost=None,
    matrix_draw_variance=False,
    matrix_variance_u=1.0,
    matrix_eigen_dirs=None,
    matrix_eigen_lens=None,
    matrix_eigen_u=0.0,
    population_fade_u=1.0,
    matrix_fade_u=1.0,
    nll_voxel_cache=None,
    nll_voxel_sweep_u=0.0,
    nll_voxel_prev_cache=None,
    nll_voxel_color_blend_u=1.0,
    nll_voxel_curv_ref_pct=None,
    nll_voxel_alpha_u=1.0,
    nll_voxel_uniform_alpha=None,
    nll_voxel_walker_w=None,
    nll_voxel_walker_clear_u=0.0,
    nll_voxel_walker_opaque_u=0.0,
    nll_voxel_sweep_wave_amp=0.0,
    nll_voxel_sweep_reveal_blend=False,
    nll_ct_bounds=None,
    nll_ct_study=None,
    nll_ct_exam=None,
    nll_ct_y=None,
    nll_ct_sweep_axis=None,
    nll_ct_plane_val=None,
    nll_ct_pivot_from=None,
    nll_ct_pivot_to=None,
    nll_ct_pivot_u=0.0,
    nll_ct_alpha_u=0.0,
    voxel_view_bounds=None,
    alt_base_study=None,
    alt_base_exam=None,
    alt_base_y=None,
    dataset_blend_u=0.0,
    highlight_classroom_w=None,
    highlight_classroom_u=0.0,
    highlight_classroom_color="#111111",
    highlight_mu_u=1.0,
    highlight_mu_w=None,
    mu_ghost_u=0.0,
    walker_w=None,
    walker_u=0.0,
    ch6_pair_line_u=0.0,
    ch6_pair_line_alpha=1.0,
    ghost_threshold_2d=None,
    ghost_threshold_2d_u=0.0,
    ghost_threshold_2d_alpha=0.42,
    ghost_threshold_2d_color=None,
    classroom_threshold_2d=None,
    classroom_threshold_2d_u=0.0,
    classroom_threshold_2d_lw=3.2,
    classroom_threshold_2d_color="#111111",
    classroom_threshold_2d_ls="--",
    show_2d_grads=False,
    grad_w_live=None,
    grad_span_frac=0.052,
    nll_grad_3d=None,
    nll_grad_3d_w=None,
    nll_grad_3d_scale=0.0,
    nll_grad_3d_u=0.0,
    nll_grad_3d_axis_lim=None,
    nll_grad_3d_is_direction=False,
    nll_grad_3d_color="#ef4444",
    nll_voxel_trail=None,
    nll_w12_surface=None,
    nll_w12_surface_u=1.0,
    nll_w12_surface_alpha=CH6_W12_PUSH_SURFACE_ALPHA,
    nll_w12_surface_reveal_origin=None,
    nll_w12_surface_reveal_u=1.0,
    nll_w12_markers_only_u=0.0,
    nll_w12_alt_surface=None,
    nll_w12_alt_surface_u=0.0,
    nll_w12_alt_surface_alpha=CH6_W12_PUSH_SURFACE_ALPHA,
    nll_w12_alt_palette="belief",
    nll_w12_marker_surface=None,
    weight_grad=None,
    weight_grad_w=None,
    weight_grad_scale=0.55,
    weight_grad_ascent=False,
    weight_grad_floor_only=False,
    weight_grad_color=CH6_VARIANCE_RED,
    surface_markers=None,
    surface_marker_s=CH6_W12_PUSH_MARKER_S,
    ch6_callouts=None,
    data_callouts=None,
    ghost_classroom_rosters=None,
    title_left=None,
    title_left_u=1.0,
    title_3d=None,
    title_3d_u=1.0,
    title_3d_scale=1.0,
    title_3d_x=0.50,
    title_3d_y=0.55,
    title_3d_color="#111111",
    title_3d_va="center",
):
    """Duo frame for sampling-variance story (continues from density_end cloud)."""
    fig, ax_data, ax3d, axes_k = _g("ch4_figure_duo_weight3d")()
    blend_u = float(np.clip(float(dataset_blend_u), 0.0, 1.0))
    if ghost_classroom_rosters:
        for gi, roster in enumerate(ghost_classroom_rosters):
            gs, ge, gy = roster
            if len(gs) > 0:
                _draw_ghost_dataset(
                    ax_data, gs, ge, gy,
                    offset=0.12 + 0.025 * float(gi),
                )
    if (
        alt_base_study is not None
        and alt_base_exam is not None
        and alt_base_y is not None
        and blend_u > 1e-6
    ):
        _draw_base_dataset_crossfade(
            ax_data,
            base_study, base_exam, base_y,
            alt_base_study, alt_base_exam, alt_base_y,
            xlim=xlim, ylim=ylim, blend_u=blend_u,
        )
    elif len(base_study) > 0:
        _draw_base_dataset(ax_data, base_study, base_exam, base_y, xlim=xlim, ylim=ylim)
    ax_data.set_xlim(*xlim)
    ax_data.set_ylim(*ylim)
    ax_data.set_xlabel("Study time (hours)", fontsize=_g("AXIS_LABEL_SIZE"), labelpad=10)
    ax_data.set_ylabel("Exam length (hours)", fontsize=_g("AXIS_LABEL_SIZE"), labelpad=10)
    ax_data.grid(alpha=0.2)

    n_g = len(ghost_ws) if ghost_ws else 0
    g_alpha = _ghost_line_alpha(n_g) * float(np.clip(ghost_fade_u, 0.0, 1.0))
    g_alpha = max(g_alpha, 0.07 * float(np.clip(ghost_fade_u, 0.0, 1.0)))
    ghost_cols = _ch6_ghost_density_colors(
        ghost_ws, W,
        reveal_u=ghost_density_reveal_u,
        z_lim=belief_z_lim if belief_z_lim is not None else axis_lim,
    ) if ghost_ws and float(ghost_density_reveal_u) > 1e-6 else None
    if ghost_ws and g_alpha > 1e-4:
        for gi, wg in enumerate(ghost_ws):
            gcol = (
                ghost_cols[gi]
                if ghost_cols is not None and gi < len(ghost_cols)
                else CH6_GHOST_COLOR
            )
            _plot_threshold(
                ax_data, wg, xlim, ylim,
                color=gcol, lw=1.2, alpha=g_alpha, zorder=3,
            )
    if mu_threshold_2d is not None and float(mu_threshold_2d_u) > 1e-6:
        _plot_threshold(
            ax_data, mu_threshold_2d, xlim, ylim,
            color=str(mu_threshold_2d_color), lw=float(mu_threshold_2d_lw),
            alpha=float(np.clip(mu_threshold_2d_u, 0.0, 1.0)), zorder=5,
        )
    if ghost_threshold_2d is not None and float(ghost_threshold_2d_u) > 1e-6:
        ghost_col = (
            str(mu_threshold_2d_color)
            if ghost_threshold_2d_color is None
            else str(ghost_threshold_2d_color)
        )
        _plot_threshold(
            ax_data, ghost_threshold_2d, xlim, ylim,
            color=ghost_col, lw=float(mu_threshold_2d_lw) * 0.9,
            alpha=float(np.clip(ghost_threshold_2d_u, 0.0, 1.0))
            * float(np.clip(ghost_threshold_2d_alpha, 0.0, 1.0)),
            zorder=4,
        )
    if classroom_threshold_2d is not None and float(classroom_threshold_2d_u) > 1e-6:
        _plot_threshold(
            ax_data, classroom_threshold_2d, xlim, ylim,
            color=str(classroom_threshold_2d_color),
            lw=float(classroom_threshold_2d_lw),
            ls=str(classroom_threshold_2d_ls),
            alpha=float(np.clip(classroom_threshold_2d_u, 0.0, 1.0)),
            zorder=6,
        )
    if (
        show_2d_grads
        and grad_w_live is not None
        and len(base_study) > 0
    ):
        G = ch6_point_nll_grad_contrib(grad_w_live, base_study, base_exam, base_y)
        _draw_point_grad_quivers_away_from_line_2d(
            ax_data, base_study, base_exam, G, grad_w_live,
            span_frac=float(grad_span_frac), compact=True,
            color=_CH6_GRAD_VECTOR_COLOR,
        )
    if data_callouts:
        for spec in data_callouts:
            if float(spec.get("u", 1.0)) <= 1e-4:
                continue
            xy = spec.get("xy")
            if xy is None:
                continue
            _draw_data_panel_callout(
                fig, ax_data, xy,
                label=str(spec.get("label", "")),
                color=str(spec.get("color", CH6_ESTIMATOR_LABEL_COLOR)),
                label_fig=spec.get("label_fig"),
                label_ha=str(spec.get("label_ha", "left")),
                label_va=str(spec.get("label_va", "center")),
                alpha=float(spec.get("u", 1.0)),
            )

    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    pop_fade = float(np.clip(population_fade_u, 0.0, 1.0))
    mat_fade = float(np.clip(matrix_fade_u, 0.0, 1.0))
    w12_surf_u = (
        float(np.clip(nll_w12_surface_u, 0.0, 1.0))
        if nll_w12_surface is not None
        else 0.0
    )
    w12_alt_u = (
        float(np.clip(nll_w12_alt_surface_u, 0.0, 1.0))
        if nll_w12_alt_surface is not None
        else 0.0
    )
    w12_markers_u = float(np.clip(nll_w12_markers_only_u, 0.0, 1.0))
    w12_panel = nll_w12_surface is not None and (
        w12_surf_u > 1e-6 or w12_alt_u > 1e-6 or w12_markers_u > 1e-6
    )
    w12_active = w12_panel
    marker_surf = (
        nll_w12_marker_surface
        if nll_w12_marker_surface is not None
        else nll_w12_surface
    )
    lik_bounds = None
    if voxel_view_bounds is not None:
        lik_bounds = tuple(voxel_view_bounds)
    elif nll_voxel_cache is not None:
        lik_bounds = tuple(nll_voxel_cache["bounds"])
    if w12_panel:
        if w12_surf_u > 1e-6 and nll_w12_surface is not None:
            surf = nll_w12_surface
        elif w12_alt_u > 1e-6 and nll_w12_alt_surface is not None:
            surf = nll_w12_alt_surface
        elif nll_w12_marker_surface is not None:
            surf = nll_w12_marker_surface
        else:
            surf = nll_w12_surface
        if view_w_st_lim is not None and view_w_el_lim is not None:
            x_lim, y_lim = view_w_st_lim, view_w_el_lim
        elif lik_bounds is not None:
            x_lim = (float(lik_bounds[0]), float(lik_bounds[1]))
            y_lim = (float(lik_bounds[2]), float(lik_bounds[3]))
        else:
            x_lim = y_lim = None
        W1 = np.asarray(surf["W1"], dtype=np.float64)
        W2 = np.asarray(surf["W2"], dtype=np.float64)
        Z = np.asarray(surf["Z"], dtype=np.float64)
        z_lo, z_hi = surf["z_lim"]
        if nll_w12_alt_surface is not None and w12_alt_u > 1e-6:
            az = np.asarray(nll_w12_alt_surface["Z"], dtype=np.float64)
            z_lo = min(float(z_lo), float(np.nanmin(az)))
            z_hi = max(float(z_hi), float(np.nanmax(az)))
        if x_lim is not None and y_lim is not None:
            inside = (
                (W1 >= float(x_lim[0])) & (W1 <= float(x_lim[1]))
                & (W2 >= float(y_lim[0])) & (W2 <= float(y_lim[1]))
            )
            if w12_surf_u > 1e-6:
                z_vis = Z[inside]
            elif nll_w12_alt_surface is not None and w12_alt_u > 1e-6:
                W1a = np.asarray(nll_w12_alt_surface["W1"], dtype=np.float64)
                W2a = np.asarray(nll_w12_alt_surface["W2"], dtype=np.float64)
                Za = np.asarray(nll_w12_alt_surface["Z"], dtype=np.float64)
                inside_a = (
                    (W1a >= float(x_lim[0])) & (W1a <= float(x_lim[1]))
                    & (W2a >= float(y_lim[0])) & (W2a <= float(y_lim[1]))
                )
                z_vis = Za[inside_a]
            else:
                z_vis = Z[inside]
            if np.any(np.isfinite(z_vis)):
                z_lo = float(np.nanmin(z_vis))
                z_hi = float(np.nanmax(z_vis))
        zpad = 0.10 * max(float(z_hi) - float(z_lo), 1e-6)
        _style_ax3d(
            ax3d, z_lim=(float(z_lo) - zpad, float(z_hi) + zpad), zlabel="NLL",
            x_lim=x_lim, y_lim=y_lim,
            elev=view_elev if view_elev is not None else ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV"))),
            azim=view_azim if view_azim is not None else float(_g("CH3_LIK_W12_CT_AZIM")),
            camera_zoom=float(camera_zoom),
        )
        rev_origin = nll_w12_surface_reveal_origin
        rev_u = float(np.clip(nll_w12_surface_reveal_u, 0.0, 1.0))
        if w12_surf_u > 1e-6:
            al = float(nll_w12_surface_alpha) * w12_surf_u
            _draw_surface(
                ax3d, surf, alpha=al, palette="nll", x_lim=x_lim, y_lim=y_lim,
                reveal_origin=rev_origin, reveal_u=rev_u,
            )
        if nll_w12_alt_surface is not None and w12_alt_u > 1e-6:
            alt = nll_w12_alt_surface
            alt_al = float(nll_w12_alt_surface_alpha) * w12_alt_u
            alt_pal = str(alt.get("palette", nll_w12_alt_palette))
            _draw_surface(
                ax3d, alt, alpha=alt_al, palette=alt_pal, x_lim=x_lim, y_lim=y_lim,
                reveal_origin=rev_origin, reveal_u=rev_u,
            )
        if x_lim is not None and y_lim is not None:
            lik_bounds = (
                float(x_lim[0]), float(x_lim[1]),
                float(y_lim[0]), float(y_lim[1]),
                -1e9, 1e9,
            )
    elif lik_bounds is not None:
        _ch6_style_nll_voxel_ax3d(
            ax3d, lik_bounds,
            elev=view_elev if view_elev is not None else ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV"))),
            azim=view_azim if view_azim is not None else float(_g("CH3_LIK_W12_CT_AZIM")),
            camera_zoom=float(camera_zoom),
        )
        if (
            nll_voxel_cache is not None
            and float(nll_voxel_alpha_u) > 1e-6
            and (
                float(nll_voxel_sweep_u) > 1e-6
                or (
                    bool(nll_voxel_sweep_reveal_blend)
                    and nll_voxel_prev_cache is not None
                    and float(nll_voxel_sweep_u) < 1.0 - 1e-9
                )
            )
        ):
            _ch6_draw_nll_voxels(
                ax3d, nll_voxel_cache, float(nll_voxel_sweep_u),
                curv_ref_pct=nll_voxel_curv_ref_pct,
                alpha_scale=float(nll_voxel_alpha_u),
                uniform_alpha=nll_voxel_uniform_alpha,
                walker_w=nll_voxel_walker_w,
                walker_clear_u=float(nll_voxel_walker_clear_u),
                walker_opaque_u=float(nll_voxel_walker_opaque_u),
                prev_cache=nll_voxel_prev_cache,
                color_blend_u=float(nll_voxel_color_blend_u),
                clip_bounds=lik_bounds,
                sweep_wave_amp=float(nll_voxel_sweep_wave_amp),
                sweep_reveal_blend=bool(nll_voxel_sweep_reveal_blend),
            )
        if nll_voxel_trail:
            _ch6_draw_nll_voxel_trail(ax3d, nll_voxel_trail)
        if (
            float(nll_ct_alpha_u) > 1e-6
            and nll_ct_study is not None
            and nll_ct_exam is not None
            and nll_ct_y is not None
            and nll_ct_bounds is not None
        ):
            ct_kw: dict[str, Any] = {}
            if nll_ct_pivot_from is not None and nll_ct_pivot_to is not None:
                ct_kw["pivot_from"] = nll_ct_pivot_from
                ct_kw["pivot_to"] = nll_ct_pivot_to
                ct_kw["pivot_u"] = float(nll_ct_pivot_u)
            elif nll_ct_sweep_axis is not None and nll_ct_plane_val is not None:
                ct_kw["sweep_axis"] = nll_ct_sweep_axis
                ct_kw["plane_val"] = float(nll_ct_plane_val)
            if ct_kw:
                _ch6_draw_nll_ct_slice(
                    ax3d,
                    nll_ct_study, nll_ct_exam, nll_ct_y, nll_ct_bounds,
                    alpha=float(nll_ct_alpha_u),
                    **ct_kw,
                )
    else:
        if view_w_st_lim is not None or view_w_el_lim is not None:
            x_lim = view_w_st_lim
            y_lim = view_w_el_lim
            z_lim = (lo, hi)
            xy_lim = None
        elif view_axis_lim is not None:
            vlo, vhi = float(view_axis_lim[0]), float(view_axis_lim[1])
            z_lim = (vlo, vhi)
            xy_lim = (vlo, vhi)
            x_lim = y_lim = None
        else:
            z_lim = (lo, hi)
            xy_lim = (lo, hi)
            x_lim = y_lim = None
        _style_ax3d(
            ax3d, z_lim=z_lim, zlabel="b", xy_lim=xy_lim,
            x_lim=x_lim, y_lim=y_lim,
            elev=view_elev, azim=view_azim,
            camera_zoom=float(camera_zoom),
        )

    W = np.asarray(W, dtype=np.float64)
    if plane_b_value is not None:
        W = W.copy()
        W[:, 2] = float(plane_b_value)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    mu_mark = (
        np.asarray(highlight_mu_w, dtype=np.float64).reshape(3)
        if highlight_mu_w is not None
        else mu
    )
    revealed_mask = np.asarray(revealed_mask, dtype=bool)
    z_for_color = belief_z_lim if belief_z_lim is not None else axis_lim
    if belief_w12:
        cols = _variance_belief_marker_colors_w12(
            W, revealed=revealed_mask, grey_u=float(grey_u), z_lim=z_for_color,
        )
        edgecolors = "none"
        linewidths = 0.0
    elif belief_z_lim is not None:
        cols = _variance_belief_marker_colors(
            W, revealed=revealed_mask, grey_u=float(grey_u), z_lim=belief_z_lim,
        )
        edgecolors = "none"
        linewidths = 0.0
    else:
        cols = _variance_marker_colors(W, revealed=revealed_mask, grey_u=float(grey_u))
        edgecolors = "white"
        linewidths = 0.35
    mas = float(np.clip(marker_alpha_scale, 0.0, 1.0)) * pop_fade
    if mas < 1.0 - 1e-6:
        cols = np.asarray(cols, dtype=np.float64).copy()
        cols[:, 3] *= mas
    if marker_bright_mask is not None:
        bright = np.asarray(marker_bright_mask, dtype=bool).reshape(-1)
        cols = np.asarray(cols, dtype=np.float64).copy()
        grey = np.array(plt.matplotlib.colors.to_rgb(CH6_VARIANCE_GREY), dtype=np.float64)
        dim = float(np.clip(marker_dim_scale, 0.05, 1.0))
        for i in range(len(cols)):
            if i < len(bright) and not bool(bright[i]):
                cols[i, :3] = cols[i, :3] * dim + grey * (1.0 - dim)
                cols[i, 3] *= 0.18 + 0.55 * dim
    if marker_color_override is not None:
        override = np.asarray(marker_color_override, dtype=np.float64)
        cols = np.asarray(cols, dtype=np.float64).copy()
        for i in range(min(len(cols), len(override))):
            if float(override[i, 3]) > 1e-4:
                cols[i] = override[i]
    draw_s = float(marker_s) * float(max(marker_s_scale, 0.5))
    bounds_mask = None
    if lik_bounds is not None:
        bounds_mask = _ch6_points_within_bounds(W, lik_bounds)
    if marker_s_per_point is not None:
        per = np.ones(len(W), dtype=np.float64)
        arr = np.asarray(marker_s_per_point, dtype=np.float64).reshape(-1)
        per[: min(len(per), len(arr))] = arr[: len(per)]
        draw_s_arr = draw_s * per
    else:
        draw_s_arr = None
    if draw_s_arr is not None:
        s_show = draw_s_arr[bounds_mask] if bounds_mask is not None else draw_s_arr
    else:
        s_show = draw_s
    voxel_overlay = nll_voxel_cache is not None
    if marker_below_voxels and voxel_overlay:
        pop_z = 1
        marker_override_z = 1
    else:
        pop_z = 20 if voxel_overlay else 8
        marker_override_z = CH6_COV_DEMO_MARKER_ZORDER
    ghost_z = 21 if voxel_overlay else 11
    mu_z = 22 if voxel_overlay else 12
    class_z = 21 if voxel_overlay else 13
    walk_z = 23 if voxel_overlay else 14
    pair_z = 19 if voxel_overlay else 10
    seg_z = 19 if voxel_overlay else 7
    if mas <= 1e-4:
        pass
    else:
        if bounds_mask is not None:
            W_show = W[bounds_mask]
            cols_show = cols[bounds_mask]
        else:
            W_show = W
            cols_show = cols
        if len(W_show) > 0:
            if w12_active:
                zz = _marker_z_on_surface(nll_w12_surface, W_show)
                ax3d.scatter(
                    W_show[:, 0], W_show[:, 1], zz,
                    s=s_show, c=cols_show, depthshade=False,
                    edgecolors=edgecolors, linewidths=linewidths, zorder=pop_z,
                )
            else:
                ax3d.scatter(
                    W_show[:, 0], W_show[:, 1], W_show[:, 2],
                    s=s_show, c=cols_show, depthshade=False,
                    edgecolors=edgecolors, linewidths=linewidths, zorder=pop_z,
                )

    def _point_in_view(pt):
        if lik_bounds is None:
            return True
        return bool(_ch6_points_within_bounds(pt, lik_bounds)[0])

    def _on_surface(pt):
        p = np.asarray(pt, dtype=np.float64).reshape(3)
        if not w12_active:
            return p
        z = float(_marker_z_on_surface(marker_surf, p.reshape(1, 3))[0])
        return np.array([p[0], p[1], z], dtype=np.float64)

    if float(mu_ghost_u) > 1e-4 and pop_fade > 1e-4 and _point_in_view(mu):
        pm = _on_surface(mu)
        ax3d.scatter(
            [pm[0]], [pm[1]], [pm[2]],
            s=88, c=CH6_VARIANCE_RED,
            alpha=0.30 * float(mu_ghost_u) * pop_fade, depthshade=False,
            edgecolors="white", linewidths=0.45, zorder=ghost_z,
        )
    elif (show_center or highlight_mu_w is not None) and pop_fade > 1e-4 and float(highlight_mu_u) > 1e-4 and _point_in_view(mu_mark):
        pm = _on_surface(mu_mark)
        ax3d.scatter(
            [pm[0]], [pm[1]], [pm[2]],
            s=95, c=CH6_VARIANCE_RED,
            alpha=float(highlight_mu_u) * pop_fade, depthshade=False,
            edgecolors="white", linewidths=0.6, zorder=mu_z,
        )

    if walker_w is not None and float(walker_u) > 1e-4 and _point_in_view(walker_w):
        ww = _on_surface(walker_w)
        ax3d.scatter(
            [ww[0]], [ww[1]], [ww[2]],
            s=92, c="#111111",
            alpha=float(walker_u) * pop_fade, depthshade=False,
            edgecolors="white", linewidths=0.55, zorder=walk_z,
        )

    if highlight_classroom_w is not None and float(highlight_classroom_u) > 1e-4 and _point_in_view(highlight_classroom_w):
        wc = _on_surface(highlight_classroom_w)
        ax3d.scatter(
            [wc[0]], [wc[1]], [wc[2]],
            s=88, c=str(highlight_classroom_color),
            alpha=float(highlight_classroom_u) * pop_fade,
            depthshade=False, edgecolors="white", linewidths=0.55, zorder=class_z,
        )

    if float(ch6_pair_line_u) > 1e-6 and float(ch6_pair_line_alpha) > 1e-4:
        wc = (
            np.asarray(highlight_classroom_w, dtype=np.float64).reshape(3)
            if highlight_classroom_w is not None
            else mu
        )
        u_seg = float(np.clip(float(ch6_pair_line_u), 0.0, 1.0))
        mid = mu + u_seg * (wc - mu)
        if w12_active:
            pts = np.vstack([mu, mid])
            zz = _marker_z_on_surface(nll_w12_surface, pts)
            ax3d.plot(
                [mu[0], mid[0]], [mu[1], mid[1]], zz,
                color=CH6_VARIANCE_RED, lw=2.0, ls=":",
                alpha=float(ch6_pair_line_alpha) * pop_fade, zorder=pair_z,
            )
        else:
            _variance_plot_seg(
                ax3d, mu, mid, color=CH6_VARIANCE_RED,
                lw=2.0, ls=":", alpha=float(ch6_pair_line_alpha) * pop_fade,
                zorder=seg_z,
            )

    if (
        not w12_active
        and nll_grad_3d is not None
        and nll_grad_3d_w is not None
        and float(nll_grad_3d_scale) > 1e-6
        and float(nll_grad_3d_u) > 1e-6
    ):
        _draw_ch6_param_grad_arrow_3d(
            ax3d, nll_grad_3d_w, nll_grad_3d,
            scale=float(nll_grad_3d_scale) * float(nll_grad_3d_u),
            alpha=float(nll_grad_3d_u) * pop_fade,
            ascent=not bool(nll_grad_3d_is_direction),
            axis_lim=nll_grad_3d_axis_lim,
            is_direction=bool(nll_grad_3d_is_direction),
            color=str(nll_grad_3d_color),
        )

    if weight_grad is not None:
        origin = weight_grad_w if weight_grad_w is not None else mu
        if origin is not None:
            _draw_weight_grad_arrow(
                ax3d, origin, weight_grad,
                surface=marker_surf if w12_panel else None,
                scale=float(weight_grad_scale),
                ascent=bool(weight_grad_ascent),
                pin_to_surface=not bool(weight_grad_floor_only),
                floor_connector=bool(weight_grad_floor_only),
                color=str(weight_grad_color),
            )

    if surface_markers and w12_panel:
        for spec in surface_markers:
            pt = np.asarray(spec["point"], dtype=np.float64).reshape(1, 3)
            if not _point_in_view(pt):
                continue
            col = str(spec.get("color", CH6_VARIANCE_RED))
            ms = float(spec.get("s", surface_marker_s))
            _draw_markers(
                ax3d, pt, color=col, s=ms,
                alpha=float(np.clip(marker_alpha_scale, 0.0, 1.0)) * pop_fade,
                z=_marker_z_on_surface(marker_surf, pt),
                edgecolors="white",
            )

    if ch6_callouts:
        active = [s for s in ch6_callouts if float(s.get("u", 1.0)) > 1e-4]
        pair_specs = [
            s for s in active
            if (s.get("pair_opposite_sides") or s.get("pair_diagonal"))
            and not s.get("label_placement")
        ]
        if len(pair_specs) == 2:
            if pair_specs[0].get("pair_diagonal") or pair_specs[1].get("pair_diagonal"):
                pl_a, pl_b = _ch6_pair_diagonal_callout_placements(
                    fig, ax3d, pair_specs[0]["point"], pair_specs[1]["point"],
                )
            else:
                pl_a, pl_b = _ch6_pair_callout_placements(
                    fig, ax3d, pair_specs[0]["point"], pair_specs[1]["point"],
                )
            for spec, pl in zip(pair_specs, (pl_a, pl_b)):
                cp = _on_surface(spec["point"])
                _draw_gaussian_callout(
                    fig, ax3d, cp,
                    label=str(spec.get("label", "")),
                    color=str(spec.get("color", CH6_VARIANCE_RED)),
                    label_fig=pl["label_fig"],
                    label_ha=pl["label_ha"],
                    label_va=pl["label_va"],
                    alpha=float(spec.get("u", 1.0)),
                )
            drawn = {id(s) for s in pair_specs}
            active = [s for s in active if id(s) not in drawn]
        for spec in active:
            pl = _ch6_callout_placement_for_spec(fig, ax3d, spec)
            if pl is None:
                continue
            cp = _on_surface(spec["point"])
            _draw_gaussian_callout(
                fig, ax3d, cp,
                label=str(spec.get("label", "")),
                color=str(spec.get("color", CH6_VARIANCE_RED)),
                label_fig=pl["label_fig"],
                label_ha=pl["label_ha"],
                label_va=pl["label_va"],
                alpha=float(spec.get("u", 1.0)),
            )

    line_color = component_color or CH6_VARIANCE_RED
    rng_color = range_color or line_color
    if axis_components:
        for axis, pairs in axis_components.items():
            al = float(pairs.get("alpha", decomp_alpha))
            for idx in pairs.get("indices", []):
                pt = W[int(idx)]
                if signed_components:
                    col = _variance_sign_color(
                        mu, pt, int(axis),
                        pos_color=CH6_VARIANCE_POS_COLOR,
                        neg_color=CH6_VARIANCE_NEG_COLOR,
                        mix_color=CH6_VARIANCE_MIX_COLOR,
                    )
                else:
                    col = pairs.get("color", line_color)
                _variance_draw_axis_component(
                    ax3d, mu, pt, int(axis), alpha=al, color=col,
                    solid_lw=float(axis_component_solid_lw),
                    dotted_lw=float(axis_component_dotted_lw),
                )
            if sign_labels and not cov_view_signs:
                _variance_draw_axis_sign_labels(
                    ax3d, mu, int(axis), axis_lim,
                    reveal_u=float(sign_labels_u) * al,
                    fontsize=float(sign_label_fontsize),
                )
    if marker_color_override is not None:
        override = np.asarray(marker_color_override, dtype=np.float64)
        hi = np.array(
            [i for i in range(min(len(W), len(override)))
             if float(override[i, 3]) > 1e-4
             and (bounds_mask is None or bool(bounds_mask[i]))],
            dtype=int,
        )
        if len(hi) > 0:
            if draw_s_arr is not None:
                hi_s = draw_s_arr[hi]
            else:
                hi_s = draw_s
            ax3d.scatter(
                W[hi, 0], W[hi, 1], W[hi, 2],
                s=hi_s, c=override[hi], depthshade=False,
                edgecolors="white", linewidths=0.45,
                zorder=marker_override_z,
            )

    if range_state:
        for axis, st in range_state.items():
            _variance_draw_centered_range(
                ax3d, mu, int(axis), float(st["lo"]), float(st["hi"]),
                alpha=float(st.get("alpha", 1.0)),
                labels=bool(st.get("labels", False)),
                lw=float(st.get("lw", 2.8)),
                color=st.get("color", rng_color),
                signed_halves=signed_range_halves,
            )

    sign_lim = sign_label_view_lim if sign_label_view_lim is not None else view_axis_lim
    st_sign = view_w_st_lim if view_w_st_lim is not None else sign_lim
    el_sign = view_w_el_lim if view_w_el_lim is not None else sign_lim
    if cov_view_signs and st_sign is not None and el_sign is not None:
        _ch6_cov_draw_view_axis_signs(
            ax3d, mu, st_sign, el_sign,
            reveal_u=float(sign_labels_u),
            fontsize=float(sign_label_fontsize),
        )
    if cov_quadrant_products and st_sign is not None and el_sign is not None:
        _ch6_cov_draw_quadrant_products(
            ax3d, mu, st_sign, el_sign, cov_quadrant_products,
            reveal_u=float(cov_quadrant_products_u),
            fontsize=max(32.0, float(sign_label_fontsize) * 0.72),
        )

    if finale_ranges:
        for axis, st in finale_ranges.items():
            _variance_draw_axis_through(
                ax3d, mu, int(axis), float(st["lo"]), float(st["hi"]),
                alpha=float(st.get("alpha", 1.0)),
                color=st.get("color", rng_color),
                signed_halves=signed_range_halves,
            )
    if show_box and box_bounds is not None and float(box_fade_u) > 1e-4:
        bf = float(np.clip(box_fade_u, 0.0, 1.0))
        _variance_draw_box(
            ax3d, box_bounds, edge_alpha=0.9 * bf, fill_alpha=0.07 * bf,
            inner_bounds=inner_bounds, gap_u=float(gap_u) * bf,
            color=box_color,
        )

    if hessian_arrow_dirs is not None and hessian_arrow_lengths is not None:
        hess_origin = (
            np.asarray(hessian_arrow_w, dtype=np.float64).reshape(3)
            if hessian_arrow_w is not None
            else mu
        )
        hess_axis_lim = (
            tuple(hessian_arrow_axis_lim)
            if hessian_arrow_axis_lim is not None
            else (float(axis_lim[0]), float(axis_lim[1]))
        )
        _draw_ch6_hessian_arrows(
            ax3d, hess_origin, hessian_arrow_dirs, hessian_arrow_lengths,
            reveal_u=float(hessian_arrow_reveal_u),
            color=str(hessian_arrow_color),
            lw=float(hessian_arrow_lw),
            axis_lim=hess_axis_lim,
        )
    if cov_arrow_dirs is not None and cov_arrow_lengths is not None:
        _draw_ch6_hessian_arrows(
            ax3d, mu, cov_arrow_dirs, cov_arrow_lengths,
            reveal_u=float(cov_arrow_reveal_u),
            color=CH6_COV_COLOR,
            lw=4.4,
        )

    if matrix_C is not None and matrix_cov_visible_cols and mat_fade > 1e-4:
        cov_u_scaled = {
            k: float(v) * mat_fade
            for k, v in (matrix_cov_u or {}).items()
        }
        var_u_scaled = None
        if matrix_var_u is not None:
            var_u_scaled = tuple(float(v) * mat_fade for v in matrix_var_u)
        _ch6_draw_morphed_cov_arrows(
            ax3d, mu, matrix_C, axis_lim=axis_lim,
            cov_u=cov_u_scaled,
            cov_visible_cols=matrix_cov_visible_cols,
            arrow_boost=matrix_arrow_boost,
            var_u=var_u_scaled,
            draw_variance=bool(matrix_draw_variance),
            variance_u=float(matrix_variance_u) * mat_fade,
            eigen_dirs=matrix_eigen_dirs,
            eigen_lens=matrix_eigen_lens,
            eigen_u=float(matrix_eigen_u) * mat_fade,
        )

    if float(cov_accum_u) > 1e-6 and float(cov_accum_length) > 1e-9:
        _ch6_cov_draw_accum_line(
            ax3d, mu, float(cov_accum_length),
            reveal_u=float(cov_accum_u), color=str(cov_accum_color),
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

    if title_left and float(title_left_u) > 1e-4:
        ax_data.text(
            0.02, 0.98, str(title_left), transform=ax_data.transAxes,
            ha="left", va="top", fontsize=22, fontweight="bold",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=float(title_left_u)),
            zorder=30,
        )

    if (
        title_3d
        and float(title_3d_u) > 1e-4
        and float(title_3d_scale) > 0.04
    ):
        sc = float(np.clip(float(title_3d_scale), 0.0, 1.0))
        fs = max(5.5, 25.0 * sc)
        edge_col = str(title_3d_color)
        ax3d.text2D(
            float(title_3d_x), float(title_3d_y), str(title_3d),
            transform=ax3d.transAxes,
            ha="center", va=str(title_3d_va), fontsize=fs, fontweight="bold",
            color="#111111",
            bbox=dict(
                boxstyle="round,pad=0.35", facecolor="white",
                edgecolor=edge_col,
                alpha=float(title_3d_u) * min(1.0, 0.25 + 0.75 * sc),
            ),
            zorder=40,
        )

    if cov_beat_caption and float(cov_beat_caption_u) > 1e-6:
        _ch6_cov_draw_beat_caption(
            fig, ax3d, cov_beat_caption, reveal_u=float(cov_beat_caption_u),
        )

    if (
        float(cov_left_fade_u) > 1e-6
        or float(cov_math_header_u) > 1e-6
        or cov_contrib_landed
        or cov_contrib_fly
        or cov_contrib_ymax is not None
    ):
        _ch6_apply_cov_math_left_panel(
            fig, ax_data, axes_k, ax3d,
            left_fade_u=float(cov_left_fade_u),
            header_u=float(cov_math_header_u),
            contrib_landed=cov_contrib_landed or [],
            contrib_fly=cov_contrib_fly,
            plane_b=float(plane_b_value) if plane_b_value is not None else 0.0,
            contrib_ymax=cov_contrib_ymax,
            header_block=cov_math_header_block,
            positive_only_pile=bool(cov_positive_only_pile),
            var_axis=int(cov_var_axis),
            var_flying=bool(cov_var_flying),
        )

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
    pin_to_surface=True, floor_z=None, floor_connector=False,
):
    """Floor-plane push arrow in (w_ST, w_EL).

    ``floor_z`` defaults to the bottom of the current 3-D z-axis (needed for NLL
    surfaces where z≈0.04 would be invisible).  When ``floor_connector`` is set,
    draw a vertical guide from the floor point up to the landscape height.
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
    zlo_ax, zhi_ax = ax3d.get_zlim()
    if floor_z is None:
        z = float(zlo_ax) + 0.02 * max(float(zhi_ax) - float(zlo_ax), 1e-6)
    else:
        z = float(floor_z)
    x1 = float(np.clip(x0 + L * d[0], lo1, hi1))
    y1 = float(np.clip(y0 + L * d[1], lo2, hi2))
    dx, dy = x1 - x0, y1 - y0
    ln = float(np.hypot(dx, dy))
    if ln < 1e-9:
        return
    ux, uy = dx / ln, dy / ln

    if (pin_to_surface or floor_connector) and surface is not None:
        z_top = float(_surface_z_at(surface, x0, y0))
        z_top += 0.01 * max(float(zhi_ax) - float(zlo_ax), 1e-6)
        ax3d.plot(
            [x0, x0], [y0, y0], [z, z_top],
            color="#1a001f", linewidth=1.2, linestyle=":",
            alpha=0.75, zorder=28,
        )
        ax3d.scatter(
            [x0], [y0], [z_top],
            s=36, c=color, edgecolors="#1a001f", linewidths=0.4,
            depthshade=False, zorder=29,
        )

    head = 0.24 * ln
    wing = 0.50 * head
    t = ln / max(L, 1e-9)
    head_lw = float(np.clip(1.2 + 2.0 * t, 1.2, 4.0))
    outline_lw = float(np.clip(2.0 + 2.8 * t, 2.0, 5.5))
    shaft_lw = float(np.clip(1.2 + 2.0 * t, 1.2, 3.8))
    dot_s = float(np.clip(28.0 + 56.0 * t, 28.0, 72.0))
    hx, hy = x1 - head * ux, y1 - head * uy
    for sx, sy in ((-uy, ux), (uy, -ux)):
        ax3d.plot(
            [x1, hx + wing * sx], [y1, hy + wing * sy], [z, z],
            color=color, linewidth=head_lw, solid_capstyle="round",
            alpha=0.98, zorder=31,
        )
    for col, width in (("#1a001f", outline_lw), (color, shaft_lw)):
        ax3d.plot(
            [x0, x1], [y0, y1], [z, z],
            color=col, linewidth=width, solid_capstyle="round",
            alpha=0.98, zorder=30,
        )
    ax3d.scatter(
        [x0], [y0], [z],
        s=dot_s,
        c=color, edgecolors="#1a001f", linewidths=0.5,
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


_CH6_WEIGHT3D_DUO_RECTS = None
_CH6_DUAL_2D_LAYOUT = None


def _ch6_weight3d_duo_layout():
    global _CH6_WEIGHT3D_DUO_RECTS
    if _CH6_WEIGHT3D_DUO_RECTS is not None:
        return _CH6_WEIGHT3D_DUO_RECTS

    def _ax_rect(ax):
        bb = ax.get_position()
        return (float(bb.x0), float(bb.y0), float(bb.width), float(bb.height))

    fig, ax_data, ax3d, axes_k = _g("ch4_figure_duo_weight3d")()
    fig.canvas.draw()
    _CH6_WEIGHT3D_DUO_RECTS = (
        _ax_rect(ax_data),
        tuple(_ax_rect(ax) for ax in axes_k),
        _ax_rect(ax3d),
    )
    plt.close(fig)
    return _CH6_WEIGHT3D_DUO_RECTS


def _ch6_draw_labeled_knobs(fig, ax_data, axes_k, ws, we, bb, *, scales=(1.0, 1.0, 1.0)):
    """w_ST / w_EL / b labeled faces (ch4 pack), not numbered ch3 dials."""
    ensure = _G.get("ch4_ensure_labeled_knob_pngs")
    if ensure is not None:
        ensure()
    knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
    _g("ch3_draw_knob_row")(
        fig, axes_k, float(ws), float(we), float(bb), "all",
        knob_rgbs, canvas_sides,
        rot_strip_deg=0.0, strip_scale=1.0,
        knob_rots=_g("ch3_k1_knob_rots_at")(float(ws), float(we), float(bb)),
        knob_scales=list(scales), ax_data=ax_data,
    )


def _ch6_place_knobs_in_rects(fig, ax_data, data_r, knob_rs, ws, we, bb):
    targets = _g("_ch4_02b_knob_target_rects")(data_r, knob_rs)
    axes_k = tuple(fig.add_axes(r) for r in targets)
    for ax in axes_k:
        ax.axis("off")
    _ch6_draw_labeled_knobs(fig, ax_data, axes_k, ws, we, bb)


def _ch6_place_knobs_wide(fig, ax_data, data_r, knob_rs, ws, we, bb, *, grow_u=1.0):
    """Wide-layout labeled knobs — fly in from legend params (ch4_02b / ch5 style)."""
    from PIL import Image

    grow_u = float(np.clip(float(grow_u), 0.0, 1.0))
    if grow_u <= 1e-4:
        return
    if grow_u >= 1.0 - 1e-6:
        _ch6_place_knobs_in_rects(fig, ax_data, data_r, knob_rs, ws, we, bb)
        return
    ensure = _G.get("ch4_ensure_labeled_knob_pngs")
    if ensure is not None:
        ensure()
    knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
    rots = _g("ch3_k1_knob_rots_at")(float(ws), float(we), float(bb))
    u = float(_g("ch3_knob_smoothstep")(grow_u))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    s1c, s2c, s3c = _g("_ch4_02b_legend_three_starts")(fig, ax_data, renderer, data_r)
    starts = (s1c, s2c, s3c)
    targets = _g("_ch4_02b_knob_target_rects")(data_r, knob_rs)
    tiny_w = max(0.006, 0.02 * (1.0 - u) + 0.001)
    for i, (sc, tgt) in enumerate(zip(starts, targets)):
        tx0, ty0, tw, th = tgt
        tcx = tx0 + 0.5 * tw
        tcy = ty0 + 0.5 * th
        sx, sy = sc
        cx = sx + u * (tcx - sx)
        cy = sy + u * (tcy - sy)
        rw = float(tiny_w + u * (tw - tiny_w))
        rh = float(tiny_w + u * (th - tiny_w))
        arr = np.asarray(
            _g("_ch3_knob_pil_rotated_square")(knob_rgbs[i], float(rots[i]), canvas_sides[i]),
            dtype=np.uint8,
        )
        _g("_ch4_02b_add_knob_rect")(
            fig, Image.fromarray(arr), cx - 0.5 * rw, cy - 0.5 * rh, rw, rh,
        )


def _ch6_place_knobs_duo_fade(fig, ax_data, ws, we, bb, *, fade_u=1.0):
    """Labeled knobs at fixed duo positions; fade opacity only (no motion)."""
    from PIL import Image

    fade_u = float(np.clip(float(fade_u), 0.0, 1.0))
    if fade_u <= 1e-4:
        return
    duo_data, duo_knobs, _ = _ch6_weight3d_duo_layout()
    ensure = _G.get("ch4_ensure_labeled_knob_pngs")
    if ensure is not None:
        ensure()
    knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
    rots = _g("ch3_k1_knob_rots_at")(float(ws), float(we), float(bb))
    targets = _g("_ch4_02b_knob_target_rects")(duo_data, duo_knobs)
    for i, tgt in enumerate(targets):
        tx0, ty0, tw, th = tgt
        arr = np.asarray(
            _g("_ch3_knob_pil_rotated_square")(knob_rgbs[i], float(rots[i]), canvas_sides[i]),
            dtype=np.uint8,
        )
        if fade_u < 1.0 - 1e-6:
            arr = arr.copy()
            arr[..., 3] = (arr[..., 3].astype(np.float64) * fade_u).astype(np.uint8)
        _g("_ch4_02b_add_knob_rect")(
            fig, Image.fromarray(arr), tx0, ty0, tw, th,
        )


def _ch6_draw_2d_reel_panel(
    ax_data,
    *,
    xlim,
    ylim,
    base_study,
    base_exam,
    base_y,
    show_base=True,
    pop_study=None,
    pop_exam=None,
    pop_y=None,
    pop_alpha=0.16,
    w_live=None,
    ghost_ws=None,
    ghost_fade_u=1.0,
    markers=None,
    ghost_density_reveal_u=0.0,
    marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
    show_legend=False,
):
    if pop_study is not None:
        _draw_population_scatter(ax_data, pop_study, pop_exam, pop_y, alpha=pop_alpha)
        ax_data.set_xlim(*xlim)
        ax_data.set_ylim(*ylim)

    if show_base and base_study is not None and len(base_study) > 0:
        _draw_base_dataset(ax_data, base_study, base_exam, base_y, xlim=xlim, ylim=ylim)
    else:
        ax_data.set_xlim(*xlim)
        ax_data.set_ylim(*ylim)
        ax_data.grid(alpha=0.2)

    ax_data.set_xlabel("Study time (hours)", fontsize=_g("AXIS_LABEL_SIZE"), labelpad=10)
    ax_data.set_ylabel("Exam length (hours)", fontsize=_g("AXIS_LABEL_SIZE"), labelpad=10)
    ax_data.grid(alpha=0.2)

    n_g = len(ghost_ws) if ghost_ws else 0
    g_alpha = _ghost_line_alpha(n_g) * float(np.clip(ghost_fade_u, 0.0, 1.0))
    ghost_cols = _ch6_ghost_density_colors(
        ghost_ws, markers,
        reveal_u=ghost_density_reveal_u,
        z_lim=marker_axis_lim,
    ) if ghost_ws and markers is not None else None
    if ghost_ws and g_alpha > 1e-4:
        for gi, wg in enumerate(ghost_ws):
            gcol = (
                ghost_cols[gi]
                if ghost_cols is not None and gi < len(ghost_cols)
                else CH6_GHOST_COLOR
            )
            _plot_threshold(
                ax_data, wg, xlim, ylim,
                color=gcol, lw=1.2, alpha=g_alpha, zorder=3,
            )
    if w_live is not None:
        _plot_threshold(
            ax_data, w_live, xlim, ylim,
            color=CH6_LINE_COLOR, lw=2.5, alpha=1.0, zorder=6,
        )
    if not show_legend:
        leg = ax_data.get_legend()
        if leg is not None:
            leg.remove()
    else:
        _g("finalize_style_legend_tex")(ax_data)


def _frame_ch6_fullscreen_2d(
    *,
    xlim,
    ylim,
    base_study,
    base_exam,
    base_y,
    show_base=True,
    pop_study=None,
    pop_exam=None,
    pop_y=None,
    pop_alpha=0.16,
    w_live=None,
    ghost_ws=None,
    ghost_fade_u=1.0,
    show_legend=False,
):
    """Full-screen wide 2D only — no knobs, no 3D (ch4_02b data slot)."""
    wide_data, _ = _g("_ch4_02b_wide_layout")()
    fig = plt.figure(figsize=_g("CH4_DUO_FIGSIZE"))
    fig.patch.set_facecolor("white")
    ax_data = fig.add_axes(wide_data)
    _ch6_draw_2d_reel_panel(
        ax_data,
        xlim=xlim, ylim=ylim,
        base_study=base_study, base_exam=base_exam, base_y=base_y,
        show_base=show_base,
        pop_study=pop_study, pop_exam=pop_exam, pop_y=pop_y, pop_alpha=pop_alpha,
        w_live=w_live, ghost_ws=ghost_ws, ghost_fade_u=ghost_fade_u,
        show_legend=show_legend,
    )
    return _fig_to_plot_wide(fig)


def _ch6_dual_2d_layout():
    """Wide fullscreen rect + equal 50/50 left/right 2D rects (no knobs)."""
    global _CH6_DUAL_2D_LAYOUT
    if _CH6_DUAL_2D_LAYOUT is not None:
        return _CH6_DUAL_2D_LAYOUT
    wide_data, _ = _g("_ch4_02b_wide_layout")()
    x0, y0, fw, fh = wide_data
    margin_l = float(x0)
    margin_r = 1.0 - float(x0) - float(fw)
    gap = 0.045
    total_w = 1.0 - margin_l - margin_r
    panel_w = (total_w - gap) / 2.0
    left_r = (margin_l, float(y0), float(panel_w), float(fh))
    right_r = (margin_l + panel_w + gap, float(y0), float(panel_w), float(fh))
    _CH6_DUAL_2D_LAYOUT = (wide_data, left_r, right_r)
    return _CH6_DUAL_2D_LAYOUT


def _ch6_reel_tick_period():
    return 1 + int(CH6_N_FLASH) + int(CH6_N_SEQ_HOLD)


@lru_cache(maxsize=1)
def _ch6_density_end_d1_pack_cached():
    return ch6_population_param_cloud_pack(
        CH6_N_CLASS_BASE,
        seed=17,
        seed_from_key="D1",
        n_reel=CH6_DENSITY_END_D1_N_REEL,
    )


def _ch6_density_end_d1_pack():
    """Shared D1 dist-reel pack for ch6_89 end state and ch6_100 handoff."""
    return _ch6_density_end_d1_pack_cached()


def _ch6_n7_roster_from_103():
    """Seven-student classroom at the end of ch6_103 (canonical six + outlier)."""
    s6, e6, y6 = _ch6_d1_balanced_six()
    return (
        np.append(s6, 5.0),
        np.append(e6, 2.0),
        np.append(y6, 0),
    )


def _ch6_marker_axis_lim_from_pack(pack, *, pad_frac=0.12):
    """Symmetric (w_ST, w_EL, b) cube containing every marker shown in a reel pack."""
    chunks = [
        np.asarray(pack["landed"], dtype=np.float64),
        np.stack([pack["w_open"], pack["w_seed"]], axis=0),
    ]
    for step in pack["reel_steps"]:
        chunks.append(np.asarray(step["markers"], dtype=np.float64))
    W = np.vstack(chunks)
    mag = float(np.abs(W).max())
    lim = float(np.ceil(mag * (1.0 + float(pad_frac)) * 2.0) / 2.0)
    lim = max(lim, 1.0)
    return (-lim, lim)


@lru_cache(maxsize=2)
def _ch6_wild_n100_reel_pack_cached(n_reel: int):
    """Wild-mixture n=100 classrooms (same population as ch6_98 right panel)."""
    return ch6_population_param_cloud_pack(
        100,
        seed=97,
        n_reel=int(n_reel),
        pop_mu_fail=CH6_WILD_MU_FAIL,
        pop_mu_pass=CH6_WILD_MU_PASS,
        pop_cov_fail=CH6_WILD_COV_FAIL,
        pop_cov_pass=CH6_WILD_COV_PASS,
        pop_seed_offset=31,
    )


@lru_cache(maxsize=2)
def _ch6_heterogeneous_n100_reel_pack_cached(n_reel: int):
    """n=100 classrooms — uniform weight spread + extreme heterogeneous mixes."""
    return ch6_population_heterogeneous_cloud_pack(
        100, seed=106, n_reel=int(n_reel), uniform_weights=True,
    )


def _ch6_growth_sequence_to_n(*, target_n=100, seed=105):
    """Grow the ch6_103 n=7 roster by drawing students from the D1 population."""
    s, e, y = _ch6_n7_roster_from_103()
    y = np.asarray(y, dtype=np.int64)
    pop_s, pop_e, pop_y = ch6_sample_population(CH6_POP_SIZE, seed=17 + 11)
    rng = np.random.default_rng(int(seed))
    n_add = int(target_n) - len(s)
    extra_ix = rng.choice(len(pop_s), size=n_add, replace=False)
    states: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        (np.asarray(s, dtype=np.float64), np.asarray(e, dtype=np.float64), y.copy()),
    ]
    for i in extra_ix:
        s = np.append(s, float(pop_s[i]))
        e = np.append(e, float(pop_e[i]))
        y = np.append(y, int(pop_y[i]))
        states.append((
            np.asarray(s, dtype=np.float64),
            np.asarray(e, dtype=np.float64),
            y.copy(),
        ))
    return states


@lru_cache(maxsize=4)
def _ch6_dense_n_sweep_states_cached(seed: int, n_min: int, n_reel: int, pad_frac: float):
    """Final-reel state per ``n`` with a dense dist reel (like ch6_103)."""
    ns = tuple(n for n in ch6_population_n_sweep_ns() if int(n) >= int(n_min))
    return _ch6_dense_n_sweep_states_build(
        seed=int(seed), n_reel=int(n_reel), pad_frac=float(pad_frac), ns=ns,
    )


def _ch6_dense_n_sweep_states_build(*, seed, n_reel, pad_frac, ns):
    """Build dense sweep states for an explicit list of classroom sizes."""
    anchor_s, anchor_e, anchor_y = _ch6_n7_roster_from_103()
    states: list[dict] = []
    weights: list[np.ndarray] = []
    ns = tuple(int(n) for n in ns)
    for ni, n_class in enumerate(ns):
        print(
            f"  dense n-sweep: n={n_class} ({ni + 1}/{len(ns)}, reel={n_reel})…",
            flush=True,
        )
        pack_kw: dict = dict(seed=int(seed), n_reel=int(n_reel))
        if int(n_class) == 7:
            pack_kw["anchor_roster"] = (anchor_s, anchor_e, anchor_y)
        pack = ch6_population_param_cloud_pack(int(n_class), **pack_kw)
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
    return (-lim, lim), tuple(states)


@lru_cache(maxsize=1)
def _ch6_d1_peel_marker_clouds_cached():
    """Final-reel marker clouds for D1 classrooms of size n=6…20."""
    clouds: dict[int, np.ndarray] = {}
    pack20 = _ch6_density_end_d1_pack_cached()
    clouds[20] = np.asarray(pack20["reel_steps"][-1]["markers"], dtype=np.float64)
    for n in range(19, 5, -1):
        pack = ch6_population_param_cloud_pack(
            n, seed=17, seed_from_key="D1",
            n_reel=ch6_population_n_sweep_n_reel(n),
        )
        clouds[n] = np.asarray(pack["reel_steps"][-1]["markers"], dtype=np.float64)
    return clouds


def _ch6_clip_weights_to_lim(W, axis_lim):
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    return np.clip(np.asarray(W, dtype=np.float64), lo, hi)


def _ch6_clip_weights_to_box(W, box_bounds):
    """Clip each axis to the variance-box faces."""
    out = np.asarray(W, dtype=np.float64).copy()
    for i in range(3):
        lo, hi = float(box_bounds[i][0]), float(box_bounds[i][1])
        out[:, i] = np.clip(out[:, i], lo, hi)
    return out


def _ch6_box_center_span(box_bounds):
    center = np.array([
        0.5 * (float(box_bounds[i][0]) + float(box_bounds[i][1])) for i in range(3)
    ], dtype=np.float64)
    span = np.array([
        float(box_bounds[i][1]) - float(box_bounds[i][0]) for i in range(3)
    ], dtype=np.float64)
    return center, span


def _ch6_match_cloud_bbox(W_ref, W):
    """Affine per-axis map so ``W`` fills the same AABB as ``W_ref``."""
    W_ref = np.asarray(W_ref, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    lo_ref, hi_ref = W_ref.min(axis=0), W_ref.max(axis=0)
    lo, hi = W.min(axis=0), W.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    span_ref = hi_ref - lo_ref
    t = (W - lo) / span
    return lo_ref + t * span_ref


def _ch6_order_match_cloud(src, tgt):
    """Reorder ``tgt`` for smoother morphs (sorted along first PC of ``src``)."""
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    c = src.mean(axis=0)
    u, _, _ = np.linalg.svd(src - c, full_matrices=False)
    pc1 = u[0] if len(u) else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    order_src = np.argsort(src @ pc1)
    order_tgt = np.argsort(tgt @ pc1)
    out = np.empty_like(tgt)
    out[order_src] = tgt[order_tgt]
    return out


def _ch6_prepare_warp_cloud(W_ref, W, axis_lim):
    """Clip, restore ``W_ref`` AABB, and order-match for stable morph endpoints."""
    W_ref = np.asarray(W_ref, dtype=np.float64)
    matched = _ch6_match_cloud_bbox(W_ref, W)
    clipped = _ch6_clip_weights_to_lim(matched, axis_lim)
    restored = _ch6_match_cloud_bbox(W_ref, clipped)
    return _ch6_order_match_cloud(W_ref, restored)


def _ch6_prepare_warp_cloud_in_box(W_ref, W, box_bounds, axis_lim):
    """Like ``_ch6_prepare_warp_cloud`` but also clips to the variance box."""
    W_ref = np.asarray(W_ref, dtype=np.float64)
    matched = _ch6_match_cloud_bbox(W_ref, W)
    clipped = _ch6_clip_weights_to_box(matched, box_bounds)
    clipped = _ch6_clip_weights_to_lim(clipped, axis_lim)
    restored = _ch6_match_cloud_bbox(W_ref, clipped)
    return _ch6_order_match_cloud(W_ref, restored)


def _ch6_warp_target_clouds_in_box(W_orig, box_bounds, axis_lim, *, seed=117):
    """Synthetic warp targets scaled to stay inside ``box_bounds``."""
    rng = np.random.default_rng(int(seed))
    W_orig = np.asarray(W_orig, dtype=np.float64)
    n = len(W_orig)
    mu, span = _ch6_box_center_span(box_bounds)
    scale = 0.82

    iso = mu + rng.normal(0.0, 0.28 * scale, size=(n, 3)) * span

    off = np.array([0.34, 0.20, 0.12], dtype=np.float64) * span * scale
    n0 = n // 2
    bimodal = np.vstack([
        mu + off + rng.normal(0.0, 0.10, (n0, 3)) * span * scale,
        mu - off + rng.normal(0.0, 0.10, (n - n0, 3)) * span * scale,
    ])
    rng.shuffle(bimodal, axis=0)

    elongated = np.column_stack([
        rng.normal(mu[0], 0.40 * span[0] * scale, n),
        rng.normal(mu[1], 0.40 * span[1] * scale, n),
        np.full(n, mu[2]) + rng.normal(0.0, 0.04 * span[2] * scale, n),
    ])

    ring_angles = rng.uniform(0.0, 2.0 * np.pi, n)
    ring_r = rng.normal(0.38 * scale, 0.06 * scale, n)
    ring = np.column_stack([
        mu[0] + ring_r * np.cos(ring_angles) * span[0],
        mu[1] + ring_r * np.sin(ring_angles) * span[1],
        mu[2] + rng.normal(0.0, 0.05 * span[2] * scale, n),
    ])

    targets = {"original": _ch6_order_match_cloud(W_orig, W_orig.copy())}
    for key, arr in (
        ("isotropic", iso),
        ("bimodal", bimodal),
        ("elongated", elongated),
        ("ring", ring),
    ):
        targets[key] = _ch6_prepare_warp_cloud_in_box(W_orig, arr, box_bounds, axis_lim)
    return targets


def _ch6_variance_end_ranges(W, *, line_color=CH6_VARIANCE_POS_COLOR):
    """Full axis ranges + box bounds at the end of the ch6_115 variance story."""
    W = np.asarray(W, dtype=np.float64)
    data_bounds = tuple((float(W[:, i].min()), float(W[:, i].max())) for i in range(3))
    inner_bounds = tuple(
        (float(np.percentile(W[:, i], 8)), float(np.percentile(W[:, i], 92))) for i in range(3)
    )
    full_ranges: dict[int, dict] = {}
    for axis_idx in range(3):
        data_lo, data_hi = data_bounds[axis_idx]
        full_ranges[axis_idx] = {
            "lo": data_lo, "hi": data_hi, "alpha": 1.0, "labels": False, "lw": 2.8,
            "color": line_color,
        }
    return full_ranges, data_bounds, inner_bounds


def _ch6_max_arrow_fit(origin, direction, lo, hi):
    """Max scalar ``t`` so ``origin + t * direction`` stays inside axis-aligned bounds."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(direction))
    if n < 1e-12:
        return 0.0
    u = direction / n
    lim = np.inf
    for k in range(3):
        if abs(u[k]) < 1e-12:
            continue
        if u[k] > 0.0:
            lim = min(lim, (hi[k] - origin[k]) / u[k])
        else:
            lim = min(lim, (lo[k] - origin[k]) / u[k])
    return float(max(lim, 0.0)) if np.isfinite(lim) else 0.0


def _ch6_hessian_eigen_arrows(mu, H, *, box_bounds, length_frac=0.68):
    """Three orthogonal Hessian eigen-directions (softest curvature → longest arrow)."""
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    lo = np.array([float(box_bounds[i][0]) for i in range(3)], dtype=np.float64)
    hi = np.array([float(box_bounds[i][1]) for i in range(3)], dtype=np.float64)
    span = hi - lo
    base = float(length_frac) * float(np.min(span))

    info = 0.5 * (H + H.T)
    evals, evecs = np.linalg.eigh(info)
    evals = np.clip(evals, 1e-12, None)

    dirs: list[np.ndarray] = []
    raw_lens: list[float] = []
    for k in range(3):
        u = np.asarray(evecs[:, k], dtype=np.float64)
        n = float(np.linalg.norm(u))
        if n < 1e-12:
            continue
        u /= n
        if _ch6_max_arrow_fit(mu, u, lo, hi) < _ch6_max_arrow_fit(mu, -u, lo, hi):
            u = -u
        dirs.append(u)
        raw_lens.append(1.0 / np.sqrt(float(evals[k])))

    raw = np.asarray(raw_lens, dtype=np.float64)
    raw /= max(float(np.max(raw)), 1e-12)
    lengths: list[float] = []
    for u, r in zip(dirs, raw):
        fit = _ch6_max_arrow_fit(mu, u, lo, hi)
        L = min(base * float(r), fit * 0.98)
        lengths.append(max(L, 0.10 * base))
    return dirs, lengths


def _ch6_cov_eigen_arrows(mu, C, *, box_bounds, length_frac=0.68):
    """Three orthogonal sample-covariance eigen-directions (σ along each axis)."""
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    C = np.asarray(C, dtype=np.float64).reshape(3, 3)
    lo = np.array([float(box_bounds[i][0]) for i in range(3)], dtype=np.float64)
    hi = np.array([float(box_bounds[i][1]) for i in range(3)], dtype=np.float64)
    span = hi - lo
    base = float(length_frac) * float(np.min(span))

    sym = 0.5 * (C + C.T)
    evals, evecs = np.linalg.eigh(sym)
    evals = np.clip(evals, 0.0, None)

    dirs: list[np.ndarray] = []
    raw_lens: list[float] = []
    for k in range(3):
        u = np.asarray(evecs[:, k], dtype=np.float64)
        n = float(np.linalg.norm(u))
        if n < 1e-12:
            continue
        u /= n
        if _ch6_max_arrow_fit(mu, u, lo, hi) < _ch6_max_arrow_fit(mu, -u, lo, hi):
            u = -u
        dirs.append(u)
        raw_lens.append(np.sqrt(float(evals[k])))

    raw = np.asarray(raw_lens, dtype=np.float64)
    raw /= max(float(np.max(raw)), 1e-12)
    lengths: list[float] = []
    for u, r in zip(dirs, raw):
        fit = _ch6_max_arrow_fit(mu, u, lo, hi)
        L = min(base * float(r), fit * 0.98)
        lengths.append(max(L, 0.10 * base))
    return dirs, lengths


def _ch6_lerp_rect(a, b, u):
    u = float(np.clip(u, 0.0, 1.0))
    return tuple(float(a[i]) + u * (float(b[i]) - float(a[i])) for i in range(4))


def _ch6_matrix_triptych_layout():
    return CH6_MATRIX_TRIPTYCH_LEFT, CH6_MATRIX_TRIPTYCH_CENTER, CH6_MATRIX_TRIPTYCH_RIGHT


def _ch6_120_end_pose():
    """Last-frame pose shared by ch6_120 and the ch6_122 handoff."""
    pack = _ch6_114_end_pack()
    end = _ch6_118_end_state(pack)
    info = end["info"]
    return {
        "info": info,
        "mu": np.asarray(end["mu"], dtype=np.float64).reshape(3),
        "W_full": np.asarray(end["W"], dtype=np.float64),
        "axis_lim": end["axis_lim"],
        "elev": float(end["elev"]),
        "azim": float(end["azim"]),
        "box_bounds": end["box_bounds"],
    }


def _ch6_matrix_contrast_cloud():
    """Full ch6_120 cloud for the matrix-contrast triptych."""
    pose = _ch6_120_end_pose()
    W = np.asarray(pose["W_full"], dtype=np.float64)
    C = np.cov(W.T, ddof=1)
    return {**pose, "W": W, "C": C}


def _ch6_122_end_matrix_state():
    """Fully explored matrix-vector state at the end of ch6_122."""
    cov_u: dict[tuple[int, int], float] = {}
    right_explored: set[tuple[int, int]] = set()
    for j in range(3):
        right_explored.add((int(j), int(j)))
        for i in range(3):
            if int(i) == int(j):
                continue
            cov_u[(int(i), int(j))] = 1.0
            right_explored.add((int(i), int(j)))
    return {
        "var_u": (1.0, 1.0, 1.0),
        "cov_u": cov_u,
        "cov_visible_cols": {0, 1, 2},
        "left_explored_cols": {0, 1, 2},
        "right_explored_cells": right_explored,
    }


def _ch6_matrix_arrow_scale(axis_lim, *, frac=CH6_MATRIX_ARROW_SCALE_FRAC, boost=None):
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    if boost is None:
        boost = CH6_MATRIX_ARROW_BOOST
    return float(boost) * float(frac) * (hi - lo)


def _ch6_matrix_arrow_head_dims(axis_lim):
    """Fixed pyramid arrowhead size in data units (independent of vector length)."""
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    span = max(hi - lo, 1e-12)
    head_len = float(CH6_MATRIX_ARROW_HEAD_LEN_FRAC) * span
    head_half = float(CH6_MATRIX_ARROW_HEAD_HALF_FRAC) * head_len
    return head_len, head_half


def _ch6_blend_hex(c0: str, c1: str, t: float) -> str:
    t = float(np.clip(t, 0.0, 1.0))
    a = np.array(plt.matplotlib.colors.to_rgb(c0), dtype=np.float64)
    b = np.array(plt.matplotlib.colors.to_rgb(c1), dtype=np.float64)
    return plt.matplotlib.colors.to_hex(a * (1.0 - t) + b * t)


def _ch6_triptych_style_b_axis_left(ax3d):
    """Place the b-axis label and ticks on the left edge of the 3D panel."""
    ax3d.zaxis.set_rotate_label(False)
    ax3d.zaxis._axinfo["juggled"] = (1, 2, 0)
    ax3d.zaxis._axinfo["axisline"]["UNHIDE"] = True
    ax3d.zaxis.set_label_coords(0.0, 0.5)
    ax3d.zaxis.set_tick_params(pad=4)


def _ch6_draw_matrix_panel(
    ax,
    C,
    *,
    title: str,
    labels=CH6_MATRIX_AXIS_LABELS,
    diagonal_only: bool = False,
    emphasize_zero_offdiag: bool = False,
    visible_cells: set[tuple[int, int]] | None = None,
    highlight_col: int | None = None,
    highlight_cell: tuple[int, int] | None = None,
    explored_cols: set[int] | None = None,
    explored_cells: set[tuple[int, int]] | None = None,
    dim_u: float = 0.28,
    panel_u: float = 1.0,
    light_highlight: bool = False,
):
    """Draw a labeled 3×3 covariance matrix grid."""
    ax.cla()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    pu = float(np.clip(panel_u, 0.0, 1.0))
    if pu <= 1e-4:
        return
    C = np.asarray(C, dtype=np.float64).reshape(3, 3)
    explored_cols = explored_cols or set()
    explored_cells = explored_cells or set()
    n = 3
    x0, x1 = 0.10, 0.94
    y1 = 0.90
    dx = (x1 - x0) / float(n)
    dy = (y1 - 0.12) / float(n)
    ax.text(
        0.5, CH6_MATRIX_PANEL_TITLE_Y, title, ha="center", va="bottom",
        fontsize=CH6_MATRIX_PANEL_TITLE_FS, fontweight="bold", color="#222",
        transform=ax.transAxes, alpha=pu,
    )
    for j, lab in enumerate(labels):
        ax.text(
            x0 + (j + 0.5) * dx, y1 + 0.02, lab,
            ha="center", va="bottom", fontsize=CH6_MATRIX_PANEL_LABEL_FS, color="#333",
            transform=ax.transAxes, alpha=pu,
        )
        ax.text(
            x0 - 0.02, y1 - (j + 0.5) * dy, lab,
            ha="right", va="center", fontsize=CH6_MATRIX_PANEL_LABEL_FS, color="#333",
            transform=ax.transAxes, alpha=pu,
        )
    for i in range(n):
        for j in range(n):
            is_offdiag = int(i) != int(j)
            if emphasize_zero_offdiag and is_offdiag:
                val = 0.0
                show = visible_cells is None or (int(i), int(j)) in visible_cells
            elif diagonal_only and is_offdiag:
                show = False
                val = 0.0
            else:
                show = visible_cells is None or (int(i), int(j)) in visible_cells
                val = float(C[int(i), int(j)])
            if not show:
                continue
            hi_col = highlight_col is not None and int(highlight_col) == int(j)
            hi_cell = (
                highlight_cell is not None
                and int(highlight_cell[0]) == int(i)
                and int(highlight_cell[1]) == int(j)
            )
            explored = (int(j) in explored_cols) or ((int(i), int(j)) in explored_cells)
            hi = hi_cell or (hi_col and highlight_cell is None)
            if hi:
                if light_highlight:
                    face = CH6_MATRIX_VAR_HIGHLIGHT_FACE
                    edge = CH6_MATRIX_VAR_HIGHLIGHT_EDGE
                    lw = 2.2
                else:
                    face = CH6_MATRIX_COV_HIGHLIGHT_FACE
                    edge = CH6_MATRIX_HIGHLIGHT_COLOR
                    lw = 2.4
            elif explored:
                if light_highlight:
                    face = CH6_MATRIX_VAR_EXPLORED_FACE
                    edge = CH6_MATRIX_VAR_EXPLORED_EDGE
                else:
                    face = CH6_MATRIX_COV_EXPLORED_FACE
                    edge = CH6_MATRIX_COV_EXPLORED_EDGE
                lw = 1.6
            elif emphasize_zero_offdiag and is_offdiag:
                face = "#f1f5f9"
                edge = "#94a3b8"
                lw = 1.1
            else:
                face = "#f8fafc"
                edge = "#cbd5e1"
                lw = 1.0
            alpha = pu * (1.0 if (hi or explored) else (1.0 - float(dim_u) * 0.55))
            rect = plt.Rectangle(
                (x0 + j * dx, y1 - (i + 1) * dy), dx, dy,
                transform=ax.transAxes,
                facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha,
            )
            ax.add_patch(rect)
            if is_offdiag and emphasize_zero_offdiag:
                body = "0"
                fs = CH6_MATRIX_PANEL_ZERO_FS
                tcol = "#64748b"
            elif int(i) == int(j):
                body = rf"$\mathrm{{Var}}$" + f"\n{val:.2f}"
                fs = CH6_MATRIX_PANEL_CELL_FS
                tcol = "#1f2937"
            else:
                body = f"{val:+.2f}"
                fs = CH6_MATRIX_PANEL_CELL_FS
                tcol = "#1f2937"
            ax.text(
                x0 + (j + 0.5) * dx, y1 - (i + 0.5) * dy, body,
                ha="center", va="center", fontsize=fs, color=tcol,
                fontweight="bold" if (emphasize_zero_offdiag and is_offdiag) else "normal",
                transform=ax.transAxes, alpha=alpha,
            )


def _ch6_matrix_var_length(C, axis_idx, *, arrow_scale, max_var):
    return arrow_scale * np.sqrt(max(float(C[int(axis_idx), int(axis_idx)]), 0.0) / max_var)


def _ch6_matrix_cov_shift_len(C, i, j, *, arrow_scale, max_var):
    return arrow_scale * float(C[int(i), int(j)]) / max(float(np.sqrt(max_var)), 1e-12)


def _ch6_arrow_perp_frame(direction):
    """Unit direction plus an orthonormal perpendicular pair."""
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    if n <= 1e-12:
        return None
    d = d / n
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(d @ ref)) > 0.92:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(d, ref)
    un = float(np.linalg.norm(u))
    if un <= 1e-12:
        return None
    u = u / un
    v = np.cross(d, u)
    return d, u, v


def _ch6_draw_segment(
    ax3d,
    p0,
    p1,
    *,
    color,
    lw=4.6,
    alpha=0.96,
    zorder=28,
    arrowhead: bool = True,
    axis_lim=None,
):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    vec = p1 - p0
    L = float(np.linalg.norm(vec))
    if L <= 1e-9:
        return
    if not arrowhead or axis_lim is None:
        ax3d.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color=str(color), linewidth=float(lw), solid_capstyle="round",
            alpha=float(alpha), zorder=float(zorder),
        )
        return
    frame = _ch6_arrow_perp_frame(vec)
    if frame is None:
        ax3d.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color=str(color), linewidth=float(lw), solid_capstyle="round",
            alpha=float(alpha), zorder=float(zorder),
        )
        return
    d, u, v = frame
    head_len, head_half = _ch6_matrix_arrow_head_dims(axis_lim)
    tip = p1
    base_c = tip - head_len * d
    if float(np.dot(base_c - p0, d)) > 1e-9:
        ax3d.plot(
            [p0[0], base_c[0]], [p0[1], base_c[1]], [p0[2], base_c[2]],
            color=str(color), linewidth=float(lw), solid_capstyle="round",
            alpha=float(alpha), zorder=float(zorder),
        )
    c0 = base_c + head_half * u + head_half * v
    c1 = base_c + head_half * u - head_half * v
    c2 = base_c - head_half * u - head_half * v
    c3 = base_c - head_half * u + head_half * v
    faces = [
        [tip, c0, c1],
        [tip, c1, c2],
        [tip, c2, c3],
        [tip, c3, c0],
        [c0, c1, c2, c3],
    ]
    coll = Poly3DCollection(
        faces,
        facecolors=str(color),
        edgecolors=str(color),
        linewidths=0.9,
        alpha=float(alpha),
    )
    coll.set_zorder(float(zorder) + 1.0)
    ax3d.add_collection3d(coll)


def _ch6_draw_segment_gradient(
    ax3d,
    p0,
    p1,
    *,
    color_start,
    color_end,
    lw=4.4,
    n_steps: int = 10,
):
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    for k in range(int(n_steps)):
        t0 = float(k) / float(n_steps)
        t1 = float(k + 1) / float(n_steps)
        q0 = p0 + t0 * (p1 - p0)
        q1 = p0 + t1 * (p1 - p0)
        col = _ch6_blend_hex(color_start, color_end, 0.5 * (t0 + t1))
        _ch6_draw_segment(ax3d, q0, q1, color=col, lw=lw)


def _ch6_matrix_column_tip(
    mu,
    C,
    j,
    *,
    arrow_scale,
    max_var,
    var_u_j: float,
    cov_u: dict[tuple[int, int], float],
    off_axes: list[int],
    cov_active: bool,
):
    """Single μ-origin tip for column *j*: axis-j variance, tilted by cov entries."""
    basis = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    Lj = _ch6_matrix_var_length(C, j, arrow_scale=arrow_scale, max_var=max_var)
    var_scale = 1.0 if cov_active else float(np.clip(var_u_j, 0.0, 1.0))
    tip = Lj * var_scale * basis[int(j)]
    for axis_i in off_axes:
        u_ij = float(np.clip(cov_u.get((int(axis_i), int(j)), 0.0), 0.0, 1.0))
        if u_ij <= 1e-6:
            continue
        shift = _ch6_matrix_cov_shift_len(
            C, axis_i, j, arrow_scale=arrow_scale, max_var=max_var,
        ) * u_ij
        if abs(shift) > 1e-9:
            tip = tip + shift * basis[int(axis_i)]
    return mu + tip


def _ch6_matrix_cov_vector_color(cov_u: dict[tuple[int, int], float], j: int, off_axes: list[int]) -> str:
    """Darker purple as more covariance tilts contribute to the column vector."""
    n_tilts = sum(
        1 for axis_i in off_axes
        if float(cov_u.get((int(axis_i), int(j)), 0.0)) > 1e-6
    )
    if n_tilts <= 0:
        return CH6_MATRIX_VAR_COLOR
    idx = min(n_tilts - 1, len(CH6_MATRIX_COV_TILT_COLORS) - 1)
    return CH6_MATRIX_COV_TILT_COLORS[idx]


def _ch6_draw_matrix_vectors(
    ax3d,
    mu,
    C,
    *,
    axis_lim,
    var_u: tuple[float, float, float] | None = None,
    cov_u: dict[tuple[int, int], float] | None = None,
    cov_visible_cols: set[int] | None = None,
    arrow_boost=None,
    draw_variance: bool = True,
):
    """Variance arrows from μ; covariance columns as single tilted μ-origin overlays."""
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    C = np.asarray(C, dtype=np.float64).reshape(3, 3)
    basis = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    arrow_scale = _ch6_matrix_arrow_scale(axis_lim, boost=arrow_boost)
    max_var = float(max(np.diag(C).max(), 1e-12))
    var_u = var_u if var_u is not None else (0.0, 0.0, 0.0)
    cov_u = cov_u or {}
    cov_visible_cols = cov_visible_cols or set()

    for j in range(3):
        off_axes = [i for i in range(3) if i != int(j)]
        Lj = _ch6_matrix_var_length(C, j, arrow_scale=arrow_scale, max_var=max_var)
        if draw_variance:
            uj = float(np.clip(var_u[j], 0.0, 1.0))
            if uj > 1e-6 and Lj > 1e-9:
                _ch6_draw_segment(
                    ax3d, mu, mu + Lj * uj * basis[j],
                    color=CH6_MATRIX_VAR_COLOR, lw=5.4, zorder=27, axis_lim=axis_lim,
                )

        cov_visible = int(j) in cov_visible_cols
        cov_active = any(float(cov_u.get((int(i), int(j)), 0.0)) > 1e-6 for i in off_axes)
        if not (cov_visible or cov_active):
            continue
        tip = _ch6_matrix_column_tip(
            mu, C, j,
            arrow_scale=arrow_scale,
            max_var=max_var,
            var_u_j=1.0,
            cov_u=cov_u,
            off_axes=off_axes,
            cov_active=True,
        )
        if np.linalg.norm(tip - mu) <= 1e-9:
            continue
        col = _ch6_matrix_cov_vector_color(cov_u, j, off_axes)
        _ch6_draw_segment(ax3d, mu, tip, color=col, lw=5.0, zorder=29, axis_lim=axis_lim)


def _ch6_lerp_cov_matrix(C0, C1, u: float):
    u = float(np.clip(u, 0.0, 1.0))
    C0 = np.asarray(C0, dtype=np.float64).reshape(3, 3)
    C1 = np.asarray(C1, dtype=np.float64).reshape(3, 3)
    return (1.0 - u) * C0 + u * C1


def _ch6_draw_morphed_cov_arrows(
    ax3d,
    mu,
    C,
    *,
    axis_lim,
    cov_u: dict[tuple[int, int], float] | None = None,
    cov_visible_cols: set[int] | None = None,
    arrow_boost=None,
    var_u: tuple[float, float, float] | None = None,
    draw_variance: bool = True,
    variance_u: float = 1.0,
    eigen_dirs=None,
    eigen_lens=None,
    eigen_u: float = 0.0,
):
    """Purple Σ-column arrows that shrink / morph into sample-covariance eigen-arrows."""
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    C = np.asarray(C, dtype=np.float64).reshape(3, 3)
    basis = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    arrow_scale = _ch6_matrix_arrow_scale(axis_lim, boost=arrow_boost)
    max_var = float(max(np.diag(C).max(), 1e-12))
    var_u = var_u if var_u is not None else (0.0, 0.0, 0.0)
    cov_u = cov_u or {}
    cov_visible_cols = cov_visible_cols or set()
    eigen_u = float(np.clip(eigen_u, 0.0, 1.0))
    var_fade = float(np.clip(variance_u, 0.0, 1.0)) * (1.0 - eigen_u)
    eig_cols = (
        CH6_MATRIX_VAR_COLOR,
        CH6_MATRIX_COV_TILT_COLORS[0],
        CH6_MATRIX_COV_TILT_COLORS[min(1, len(CH6_MATRIX_COV_TILT_COLORS) - 1)],
    )

    if draw_variance and var_fade > 1e-6:
        for j in range(3):
            Lj = _ch6_matrix_var_length(C, j, arrow_scale=arrow_scale, max_var=max_var)
            uj = float(np.clip(var_u[j], 0.0, 1.0)) * var_fade
            if uj > 1e-6 and Lj > 1e-9:
                _ch6_draw_segment(
                    ax3d, mu, mu + Lj * uj * basis[j],
                    color=CH6_MATRIX_VAR_COLOR, lw=5.4, zorder=27, axis_lim=axis_lim,
                )

    for k, j in enumerate(sorted(int(x) for x in cov_visible_cols)):
        off_axes = [i for i in range(3) if i != int(j)]
        col_tip = _ch6_matrix_column_tip(
            mu, C, j,
            arrow_scale=arrow_scale,
            max_var=max_var,
            var_u_j=1.0,
            cov_u=cov_u,
            off_axes=off_axes,
            cov_active=True,
        )
        col_color = _ch6_matrix_cov_vector_color(cov_u, j, off_axes)
        if (
            eigen_dirs is not None
            and eigen_lens is not None
            and int(k) < len(eigen_dirs)
            and eigen_u > 1e-6
        ):
            direction = np.asarray(eigen_dirs[int(k)], dtype=np.float64).reshape(3)
            length = float(eigen_lens[int(k)])
            eig_tip = mu + length * direction
            tip = mu + (1.0 - eigen_u) * (col_tip - mu) + eigen_u * (eig_tip - mu)
            end_col = eig_cols[min(int(k), len(eig_cols) - 1)]
            col = _ch6_blend_hex(col_color, end_col, eigen_u)
        else:
            tip = col_tip
            col = col_color
        if np.linalg.norm(tip - mu) <= 1e-9:
            continue
        _ch6_draw_segment(ax3d, mu, tip, color=col, lw=5.0, zorder=29, axis_lim=axis_lim)


def _ch6_triptych_draw_cloud(
    ax3d, W, mu, *, axis_lim, elev, azim, marker_s=28.0,
    revealed_mask=None, grey_u=0.0, triptych_b_axis=True,
    belief_z_lim=None,
):
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    _style_ax3d(
        ax3d, z_lim=(lo, hi), xy_lim=(lo, hi), zlabel=r"$b$",
        elev=float(elev), azim=float(azim), minimal_ui=False,
    )
    if triptych_b_axis:
        _ch6_triptych_style_b_axis_left(ax3d)
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    if revealed_mask is None:
        revealed = np.ones(len(W), dtype=bool)
    else:
        revealed = np.asarray(revealed_mask, dtype=bool)
    if belief_z_lim is not None:
        cols = _variance_belief_marker_colors(
            W, revealed=revealed, grey_u=float(grey_u), z_lim=belief_z_lim,
        )
        edgecolors = "none"
        linewidths = 0.0
    else:
        cols = _variance_marker_colors(W, revealed=revealed, grey_u=float(grey_u))
        edgecolors = "white"
        linewidths = 0.35
    ax3d.scatter(
        W[:, 0], W[:, 1], W[:, 2],
        s=float(marker_s), c=cols, depthshade=False,
        edgecolors=edgecolors, linewidths=linewidths, zorder=8,
    )
    ax3d.scatter(
        [mu[0]], [mu[1]], [mu[2]],
        s=95, c=CH6_VARIANCE_RED, alpha=1.0, depthshade=False,
        edgecolors="white", linewidths=0.6, zorder=12,
    )


def _frame_ch6_cov_matrix_triptych(
    *,
    W,
    mu,
    C,
    axis_lim,
    elev,
    azim,
    marker_s=28.0,
    layout_u=1.0,
    panel_u=1.0,
    left_highlight_col: int | None = None,
    left_highlight_cell: tuple[int, int] | None = None,
    right_highlight_col: int | None = None,
    right_highlight_cell: tuple[int, int] | None = None,
    left_explored_cols: set[int] | None = None,
    right_explored_cells: set[tuple[int, int]] | None = None,
    right_dim_u: float = 0.0,
    var_u: tuple[float, float, float] | None = None,
    cov_u: dict[tuple[int, int], float] | None = None,
    cov_visible_cols: set[int] | None = None,
    arrow_boost=None,
    draw_variance: bool = True,
    revealed_mask=None,
    grey_u: float = 0.0,
    triptych_b_axis: bool = True,
    eigen_dirs=None,
    eigen_lens=None,
    eigen_u: float = 0.0,
    variance_u: float = 1.0,
    belief_z_lim=None,
):
    """Left: diagonal variances | center: 3D cloud + vectors | right: full Σ."""
    fig = plt.figure(figsize=_g("CH4_DUO_FIGSIZE"))
    fig.patch.set_facecolor("white")
    left_r, center_r, right_r = _ch6_matrix_triptych_layout()
    _, _, duo_3d = _ch6_weight3d_duo_layout()
    lu = float(np.clip(layout_u, 0.0, 1.0))
    center_rect = _ch6_lerp_rect(duo_3d, center_r, lu)
    side_u = float(np.clip((lu - 0.18) / 0.82, 0.0, 1.0)) * float(np.clip(panel_u, 0.0, 1.0))

    ax3d = fig.add_axes(center_rect, projection="3d")
    _ch6_triptych_draw_cloud(
        ax3d, W, mu, axis_lim=axis_lim, elev=elev, azim=azim, marker_s=marker_s,
        revealed_mask=revealed_mask, grey_u=float(grey_u), triptych_b_axis=triptych_b_axis,
        belief_z_lim=belief_z_lim,
    )
    _ch6_draw_morphed_cov_arrows(
        ax3d, mu, C, axis_lim=axis_lim,
        cov_u=cov_u, cov_visible_cols=cov_visible_cols,
        arrow_boost=arrow_boost, var_u=var_u, draw_variance=draw_variance,
        variance_u=float(variance_u),
        eigen_dirs=eigen_dirs, eigen_lens=eigen_lens, eigen_u=float(eigen_u),
    )

    C = np.asarray(C, dtype=np.float64).reshape(3, 3)
    diag = np.diag(np.diag(C))

    if side_u > 1e-4:
        ax_left = fig.add_axes(left_r)
        _ch6_draw_matrix_panel(
            ax_left, diag,
            title="Variance",
            emphasize_zero_offdiag=True,
            highlight_col=left_highlight_col,
            highlight_cell=left_highlight_cell,
            explored_cols=left_explored_cols,
            panel_u=side_u,
            light_highlight=True,
        )
        ax_right = fig.add_axes(right_r)
        _ch6_draw_matrix_panel(
            ax_right, C,
            title="Covariance",
            highlight_col=right_highlight_col,
            highlight_cell=right_highlight_cell,
            explored_cells=right_explored_cells,
            dim_u=float(right_dim_u),
            panel_u=side_u,
        )

    return _fig_to_plot(fig)


def _draw_ch6_hessian_arrows(
    ax3d,
    origin,
    dirs,
    lengths,
    *,
    reveal_u=1.0,
    color=CH6_HESSIAN_ARROW_COLOR,
    lw=3.0,
    axis_lim=None,
):
    u = float(np.clip(reveal_u, 0.0, 1.0))
    if u <= 1e-6:
        return
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    if axis_lim is None:
        axis_lim = (0.0, 1.0)
    for direction, length in zip(dirs, lengths):
        direction = np.asarray(direction, dtype=np.float64).reshape(3)
        L = float(length) * u
        if L <= 1e-6:
            continue
        tip = origin + L * direction
        _ch6_draw_segment(
            ax3d, origin, tip,
            color=str(color), lw=float(lw), alpha=0.96, zorder=30,
            arrowhead=True, axis_lim=axis_lim,
        )


def _ch6_cov_draw_accum_line(ax3d, mu, length, *, reveal_u=1.0, color=CH6_COV_COLOR):
    """Growing total Cov(w_ST,w_EL) built from point contributions (Σ Δw_ST·Δw_EL)."""
    u = float(np.clip(reveal_u, 0.0, 1.0))
    if u <= 1e-6 or float(length) <= 1e-9:
        return
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    direction = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction[:2])), 1e-12)
    L = float(length) * u
    tip = mu.copy()
    tip[0] += direction[0] * L
    tip[1] += direction[1] * L
    ax3d.plot(
        [mu[0], tip[0]], [mu[1], tip[1]], [mu[2], tip[2]],
        color=str(color), linewidth=5.2, solid_capstyle="round",
        alpha=0.96, zorder=26,
    )


def _ch6_cov_w12_ranges(W):
    """Signed range segments for w_ST and w_EL through the cloud mean."""
    W = np.asarray(W, dtype=np.float64)
    ranges: dict[int, dict] = {}
    for axis in (0, 1):
        ranges[axis] = {
            "lo": float(W[:, axis].min()),
            "hi": float(W[:, axis].max()),
            "alpha": 1.0,
            "labels": False,
            "lw": float(CH6_COV_RANGE_LW),
        }
    return ranges


def _ch6_left_panel_rect(ax_data, axes_k):
    boxes = [ax_data.get_position()] + [ax.get_position() for ax in axes_k]
    x0 = min(float(b.x0) for b in boxes)
    y0 = min(float(b.y0) for b in boxes)
    x1 = max(float(b.x0) + float(b.width) for b in boxes)
    y1 = max(float(b.y0) + float(b.height) for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def _ch6_cov_math_line_progress(n_lines: int, write_u: float) -> dict[int, float]:
    write_u = float(np.clip(write_u, 0.0, 1.0))
    per = write_u * float(max(int(n_lines), 1))
    return {i: float(np.clip(per - float(i), 0.0, 1.0)) for i in range(int(n_lines))}


def _ch6_strip_left_data_ui(ax_data, axes_k):
    """Remove 2D axis labels/ticks/spines (exam length clutter under math panel)."""
    ax_data.set_xlabel("")
    ax_data.set_ylabel("")
    ax_data.set_xticks([])
    ax_data.set_yticks([])
    ax_data.tick_params(
        axis="both", which="both",
        left=False, bottom=False, labelleft=False, labelbottom=False,
    )
    for sp in ax_data.spines.values():
        sp.set_visible(False)
    ax_data.grid(False)
    for axk in axes_k:
        axk.set_xticks([])
        axk.set_yticks([])
        axk.tick_params(
            axis="both", which="both",
            left=False, bottom=False, labelleft=False, labelbottom=False,
        )
        for sp in axk.spines.values():
            sp.set_visible(False)


def _ch6_fade_left_panel_artists(ax_data, axes_k, fade_u: float):
    """Fade the 2D scatter plot and all knob axes together."""
    vis = max(0.0, 1.0 - float(np.clip(fade_u, 0.0, 1.0)))
    for ax in [ax_data, *list(axes_k)]:
        ax.patch.set_alpha(vis)
        for artist in ax.get_children():
            if hasattr(artist, "set_alpha"):
                try:
                    artist.set_alpha(vis)
                except (TypeError, ValueError):
                    pass


def _ch6_cov_math_header_block():
    fb = _g("_ch4_formula_hand_block")
    base = dict(
        align="left",
        text_x_frac=0.04,
        block_fs=CH6_COV_MATH_FS,
        fit_to_column_width=False,
        force_wrap=False,
        mathtext_max_line_frac=0.98,
        block_fs_min=14.0,
        line_dy_pt=24.0,
        pt_units=True,
        weight=0.28,
        bold_lhs=True,
    )
    return fb(
        r"$\mathrm{Cov}(w_{\mathrm{ST}}, w_{\mathrm{EL}})"
        r"= \frac{1}{n}\,\sum_{i=1}^{n}"
        rf"({CH6_MU_WST_TEX}-w_{{\mathrm{{ST}},i}})"
        rf"({CH6_MU_WEL_TEX}-w_{{\mathrm{{EL}},i}})$",
        **base,
    )


def _ch6_cov_contrib_spec(W, mu, idx, *, scale: float):
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    prod = float(W[int(idx), 0] - mu[0]) * float(W[int(idx), 1] - mu[1])
    h = abs(prod) * float(scale)
    sign = 1.0 if prod >= 0.0 else -1.0
    col = CH6_VARIANCE_POS_COLOR if sign > 0.0 else CH6_VARIANCE_NEG_COLOR
    return max(h, 1e-9), col, sign


def _ch6_cov_contrib_scale(W, mu, indices, *, block_h_max=CH6_COV_BLOCK_H_MAX):
    """Fixed scale: tallest block has height ``block_h_max``; others proportional."""
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    peak = 0.0
    for idx in indices:
        prod = float(W[int(idx), 0] - mu[0]) * float(W[int(idx), 1] - mu[1])
        peak = max(peak, abs(prod))
    return float(block_h_max) / max(peak, 1e-9)


def _ch6_cov_pile_plan(W, mu, indices, *, block_h_max=CH6_COV_BLOCK_H_MAX):
    """Precompute proportional block scale and symmetric ±ymax for a beat."""
    scale = _ch6_cov_contrib_scale(W, mu, indices, block_h_max=block_h_max)
    pos_stack = neg_stack = 0.0
    for idx in indices:
        h, _, sign = _ch6_cov_contrib_spec(W, mu, int(idx), scale=scale)
        if float(sign) > 0.0:
            pos_stack += float(h)
        else:
            neg_stack += float(h)
    ymax = max(pos_stack, neg_stack, 0.06) * 1.06
    return float(scale), float(ymax)


def _ch6_cov_point_prod(W, mu, idx):
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    return float(W[int(idx), 0] - mu[0]) * float(W[int(idx), 1] - mu[1])


def _ch6_cov_stack_totals(W, mu, indices, *, scale: float):
    pos = neg = 0.0
    for idx in indices:
        h, _, sign = _ch6_cov_contrib_spec(W, mu, int(idx), scale=scale)
        if float(sign) > 0.0:
            pos += float(h)
        else:
            neg += float(h)
    return pos, neg


def _ch6_cov_pick_skewed_demo(W, mu, skew, *, n=40, seed=119):
    """Equal per-quadrant counts; within-quadrant picks skew the signed pile totals."""
    demo = _ch6_cov_pick_demo(W, mu, n_each=max(int(n) * 2, 60), seed=seed)
    keys = ("pp", "nn", "pn", "np")
    n = int(n)
    each = max(1, n // 4)
    rem = n - 4 * each
    counts = {keys[i]: each + (1 if rem > i else 0) for i in range(4)}

    def _pick_one(trial_seed: int):
        rng = np.random.default_rng(int(trial_seed))
        picked: list[int] = []
        for key in keys:
            pool = list(demo[key])
            if len(pool) < counts[key]:
                pool = pool + list(demo[key])
            scored = [(i, _ch6_cov_point_prod(W, mu, i)) for i in pool]
            if skew == "neg_dominant":
                scored.sort(key=lambda t: t[1])
            elif skew == "pos_dominant":
                if key in ("pp", "nn"):
                    scored.sort(key=lambda t: t[1], reverse=True)
                else:
                    scored.sort(key=lambda t: abs(t[1]))
            else:
                scored.sort(key=lambda t: abs(t[1]), reverse=True)
                mid = len(scored) // 2
                lo = max(0, mid - counts[key] // 2)
                scored = scored[lo: lo + counts[key] * 3]
            take = [i for i, _ in scored[: counts[key]]]
            if len(take) < counts[key]:
                extra = [i for i in pool if i not in take]
                rng.shuffle(extra)
                take.extend(extra[: counts[key] - len(take)])
            picked.extend(take[: counts[key]])
        rng.shuffle(picked)
        return picked[:n]

    best = _pick_one(seed + 31)
    best_gap = -1.0
    for trial in range(24):
        cand = _pick_one(seed + 31 + trial * 17)
        scale = _ch6_cov_contrib_scale(W, mu, cand)
        pos, neg = _ch6_cov_stack_totals(W, mu, cand, scale=scale)
        if skew == "neg_dominant":
            gap = neg - pos
            ok = gap > 0.0
        elif skew == "pos_dominant":
            gap = pos - neg
            ok = gap > 0.0
        else:
            gap = -abs(pos - neg)
            ok = True
        if ok and gap > best_gap:
            best, best_gap = cand, gap
    return best


def _ch6_cov_pick_equal_quadrants(W, mu, *, n=12, seed=119):
    return _ch6_cov_pick_skewed_demo(W, mu, "balanced", n=n, seed=seed)


def _ch6_ax3d_point_to_fig(fig, ax3d, pt):
    from mpl_toolkits.mplot3d import proj3d

    fig.canvas.draw()
    pt = np.asarray(pt, dtype=np.float64).reshape(3)
    x2, y2, _ = proj3d.proj_transform(
        float(pt[0]), float(pt[1]), float(pt[2]),
        ax3d.get_proj(),
    )
    xdisp, ydisp = ax3d.transData.transform((x2, y2))
    xfig, yfig = fig.transFigure.inverted().transform((xdisp, ydisp))
    return float(xfig), float(yfig)


def _ch6_ax_data_to_fig(fig, ax, x, y):
    xdisp, ydisp = ax.transData.transform((float(x), float(y)))
    xfig, yfig = fig.transFigure.inverted().transform((xdisp, ydisp))
    return float(xfig), float(yfig)


def _ch6_fig_cm_to_frac(fig, cm: float) -> float:
    return float(cm) / 2.54 / max(float(fig.get_figheight()), 1e-6)


def _ch6_cov_panel_hrect(rect):
    """Horizontally center the formula box and pile plot (same width)."""
    x0 = float(rect[0]) + CH6_COV_PANEL_BOX_X * float(rect[2])
    w = CH6_COV_PANEL_BOX_W * float(rect[2])
    return x0, w


def _ch6_cov_pile_geom():
    # Keep contribution block size; center bar in the matched-width panel.
    bar_w = 0.54 * (0.88 / CH6_COV_PANEL_BOX_W)
    x0 = 0.5 * (1.0 - bar_w)
    return x0, bar_w


def _ch6_cov_signed_ymax(blocks, *, fly=None):
    pos = neg = 0.0
    for item in blocks:
        h, _, sign = item[0], item[1], item[2]
        if float(sign) > 0.0:
            pos += float(h)
        else:
            neg += float(h)
    if fly is not None:
        fh, fsign = float(fly[1]), float(fly[3])
        if fsign > 0.0:
            pos += fh
        else:
            neg += fh
    return max(pos, neg, 0.06) * 1.22


def _ch6_cov_draw_signed_pile(ax, blocks, *, ymax):
    ym = float(max(ymax, 0.06))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-ym, ym)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.9)
        sp.set_edgecolor("#9ca3af")
    ax.axhline(0.0, color="#374151", lw=1.4, zorder=3)
    ax.grid(axis="y", alpha=0.12)
    x0, bar_w = _ch6_cov_pile_geom()
    pos_cum = 0.0
    neg_cum = 0.0
    for h, col, sign in blocks:
        hh = float(h)
        if hh <= 1e-9:
            continue
        if float(sign) > 0.0:
            ax.add_patch(
                plt.Rectangle(
                    (x0, pos_cum), bar_w, hh,
                    facecolor=col, edgecolor="#374151", linewidth=0.9,
                    alpha=0.94, zorder=4,
                )
            )
            pos_cum += hh
        else:
            ax.add_patch(
                plt.Rectangle(
                    (x0, neg_cum - hh), bar_w, hh,
                    facecolor=col, edgecolor="#374151", linewidth=0.9,
                    alpha=0.94, zorder=4,
                )
            )
            neg_cum -= hh


def _ch6_cov_draw_flying_contrib(
    fig, ax3d, ax_pile, *, W, mu, idx, height, color, sign,
    pos_cum, neg_cum, t_fly, ymax, plane_b,
):
    smooth = _g("ch3_knob_smoothstep")
    u = float(smooth(float(t_fly)))
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    pt = np.array([W[int(idx), 0], W[int(idx), 1], float(plane_b)], dtype=np.float64)
    sx, sy = _ch6_ax3d_point_to_fig(fig, ax3d, pt)
    x0, bar_w = _ch6_cov_pile_geom()
    ym = float(max(ymax, 0.06))
    h_vis = float(height) * u
    if float(sign) > 0.0:
        y_target = float(pos_cum) + h_vis * 0.5
    else:
        y_target = float(neg_cum) - h_vis * 0.5
    tx, ty = _ch6_ax_data_to_fig(fig, ax_pile, x0 + bar_w * 0.5, y_target)
    fw1 = bar_w * ax_pile.get_position().width
    fh1 = (h_vis / (2.0 * ym)) * ax_pile.get_position().height
    fw0 = fh0 = 0.003
    fw = fw0 + u * (fw1 - fw0)
    fh = fh0 + u * (fh1 - fh0)
    cx = sx + u * (tx - sx)
    cy = sy + u * (ty - sy)
    fig.patches.append(
        plt.Rectangle(
            (cx - fw * 0.5, cy - fh * 0.5), fw, fh,
            transform=fig.transFigure,
            facecolor=color, edgecolor="#374151", linewidth=0.9,
            alpha=0.96, zorder=520,
        )
    )


def _ch6_cov_draw_beat_caption(fig, ax3d, caption, *, reveal_u=1.0):
    """Caption below the 3D panel, centered under the w_ST axis label."""
    u = float(np.clip(reveal_u, 0.0, 1.0))
    if u <= 1e-6 or not caption:
        return
    fig.canvas.draw()
    bb = ax3d.get_position()
    y = bb.y0 - 0.018 + _ch6_fig_cm_to_frac(fig, CH6_COV_BEAT_CAPTION_UP_CM)
    fig.text(
        bb.x0 + 0.52 * bb.width,
        y,
        str(caption),
        transform=fig.transFigure,
        fontsize=float(_g("AXIS_LABEL_SIZE")) * CH6_COV_BEAT_CAPTION_FS_SCALE,
        color="#1f2937",
        ha="center",
        va="top",
        alpha=0.96 * u,
        zorder=60,
    )


def _ch6_cov_landed_from_indices(W, mu, indices, *, scale):
    return [_ch6_cov_contrib_spec(W, mu, i, scale=scale) for i in indices]


def _ch6_var_math_header_block(kind, *, axis_idx: int = 0):
    """Variance-story header formulas (reuse ch6_120 left-panel layout)."""
    fb = _g("_ch4_formula_hand_block")
    base = dict(
        align="center",
        text_x_frac=0.5,
        block_fs=CH6_VARIANCE_MATH_FS,
        valign="center",
        fit_to_column_width=False,
        force_wrap=True,
        mathtext_max_line_frac=0.94 * CH6_COV_PANEL_BOX_W,
        block_fs_min=12.0,
        line_dy_pt=24.0,
        pt_units=True,
        weight=0.28,
        bold_lhs=True,
        box_pad_top_pt=CH6_VARIANCE_MATH_BOX_PAD_TOP_PT,
        box_pad_bottom_pt=CH6_VARIANCE_MATH_BOX_PAD_BOTTOM_PT,
    )
    axis_specs = {
        0: (r"w_{\mathrm{ST}}", CH6_MU_WST_TEX, r"w_{\mathrm{ST},i}"),
        1: (r"w_{\mathrm{EL}}", CH6_MU_WEL_TEX, r"w_{\mathrm{EL},i}"),
        2: ("b", r"\mu_b", r"b_i"),
    }
    w, mu_s, wi = axis_specs.get(int(axis_idx), axis_specs[0])
    if kind == "variance":
        tex = (
            rf"$\mathrm{{Var}}({w})"
            rf"= \frac{{1}}{{n}}\sum_{{i=1}}^{{n}}"
            rf"({wi}-{mu_s})^2$"
        )
        base.update(fit_to_column_width=True)
    elif kind == "deviation_st":
        tex = rf"$(w_{{\mathrm{{ST}},i}}-{CH6_MU_WST_TEX})$"
    elif kind == "deviation_el":
        tex = rf"$(w_{{\mathrm{{EL}},i}}-{CH6_MU_WEL_TEX})$"
    elif kind == "mean_zero_el":
        tex = (
            r"$\frac{1}{n}\sum_{i=1}^{n}"
            rf"(w_{{\mathrm{{EL}},i}}-{CH6_MU_WEL_TEX})=0$"
        )
        base.update(
            fit_to_column_width=True,
        )
    else:
        tex = (
            rf"$\mathrm{{Var}}({w})"
            rf"= \frac{{1}}{{n}}\sum_{{i=1}}^{{n}}"
            rf"({wi}-{mu_s})^2$"
        )
    return fb(tex, **base)


def _ch6_var_contrib_spec(W, mu, idx, axis, *, scale: float, positive_only: bool = False):
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    dev = float(W[int(idx), int(axis)] - mu[int(axis)])
    if positive_only:
        h = dev * dev * float(scale)
        return max(h, 1e-9), CH6_VARIANCE_POS_COLOR, 1.0
    sign = 1.0 if dev >= 0.0 else -1.0
    h = abs(dev) * float(scale)
    col = CH6_VARIANCE_POS_COLOR if sign > 0.0 else CH6_VARIANCE_NEG_COLOR
    return max(h, 1e-9), col, sign


def _ch6_var_contrib_scale(W, mu, indices, axis, *, positive_only=False, block_h_max=CH6_COV_BLOCK_H_MAX):
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    peak = 0.0
    for idx in indices:
        dev = float(W[int(idx), int(axis)] - mu[int(axis)])
        peak = max(peak, dev * dev if positive_only else abs(dev))
    return float(block_h_max) / max(peak, 1e-9)


def _ch6_var_pile_plan(W, mu, indices, axis, *, positive_only=False, block_h_max=CH6_COV_BLOCK_H_MAX):
    scale = _ch6_var_contrib_scale(
        W, mu, indices, axis, positive_only=positive_only, block_h_max=block_h_max,
    )
    if positive_only:
        total = 0.0
        for idx in indices:
            h, _, _ = _ch6_var_contrib_spec(
                W, mu, int(idx), axis, scale=scale, positive_only=True,
            )
            total += float(h)
        ymax = max(total, 0.06) * 1.06
        return float(scale), float(ymax)
    pos_stack = neg_stack = 0.0
    for idx in indices:
        h, _, sign = _ch6_var_contrib_spec(W, mu, int(idx), axis, scale=scale, positive_only=False)
        if float(sign) > 0.0:
            pos_stack += float(h)
        else:
            neg_stack += float(h)
    ymax = max(pos_stack, neg_stack, 0.06) * 1.06
    return float(scale), float(ymax)


def _ch6_var_landed_from_indices(W, mu, indices, axis, *, scale, positive_only=False):
    return [
        _ch6_var_contrib_spec(W, mu, i, axis, scale=scale, positive_only=positive_only)
        for i in indices
    ]


def _ch6_var_positive_ymax(blocks, *, fly=None):
    total = sum(float(b[0]) for b in blocks)
    if fly is not None:
        total += float(fly[1])
    return max(total, 0.06) * 1.06


def _ch6_var_stack_totals(W, mu, indices, axis, *, scale, positive_only=False):
    pos = neg = 0.0
    for idx in indices:
        h, _, sign = _ch6_var_contrib_spec(
            W, mu, int(idx), axis, scale=scale, positive_only=positive_only,
        )
        if positive_only or float(sign) > 0.0:
            pos += float(h)
        else:
            neg += float(h)
    return pos, neg


def _ch6_var_pick_demo(W, mu, axis, *, n=40, seed=115):
    """Balanced +/- demo subset (ch6_120-style count); mean deviation ≈ 0."""
    rng = np.random.default_rng(int(seed))
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    ax = int(axis)
    dev = W[:, ax] - mu[ax]
    pos_idx = np.flatnonzero(dev > 0.0)
    neg_idx = np.flatnonzero(dev < 0.0)
    n = int(n)
    n_each = max(1, n // 2)
    if len(pos_idx) < n_each or len(neg_idx) < n_each:
        n_each = max(1, min(len(pos_idx), len(neg_idx), n // 2))

    def _rank_pool(idxs, *, reverse=True):
        ranked = sorted(int(i) for i in idxs)
        ranked.sort(key=lambda i: abs(float(dev[i])), reverse=reverse)
        return ranked

    pos_rank = _rank_pool(pos_idx)
    neg_rank = _rank_pool(neg_idx)

    def _pick_trial(trial_seed: int):
        rng_t = np.random.default_rng(int(trial_seed))
        pos_cands = pos_rank[: min(len(pos_rank), n_each * 4)]
        neg_cands = neg_rank[: min(len(neg_rank), n_each * 4)]
        rng_t.shuffle(pos_cands)
        rng_t.shuffle(neg_cands)
        picked: list[int] = []
        sum_dev = 0.0
        pos_left = list(pos_cands)
        neg_left = list(neg_cands)
        for _ in range(n_each):
            if not pos_left or not neg_left:
                break
            best_pair = None
            best_gap = float("inf")
            pos_scan = pos_left[: min(24, len(pos_left))]
            neg_scan = neg_left[: min(24, len(neg_left))]
            for p in pos_scan:
                for nn in neg_scan:
                    gap = abs(sum_dev + float(dev[p]) + float(dev[nn]))
                    if gap < best_gap:
                        best_gap = gap
                        best_pair = (p, nn)
            if best_pair is None:
                break
            p, nn = best_pair
            picked.extend([p, nn])
            sum_dev += float(dev[p]) + float(dev[nn])
            pos_left.remove(p)
            neg_left.remove(nn)
        rng_t.shuffle(picked)
        return picked[: n_each * 2]

    best = _pick_trial(seed + 31)
    best_score = float("inf")
    for trial in range(24):
        cand = _pick_trial(seed + 31 + trial * 17)
        if len(cand) < 2:
            continue
        mean_dev = abs(float(np.mean([dev[i] for i in cand])))
        scale = _ch6_var_contrib_scale(W, mu, cand, ax, positive_only=False)
        pos_h, neg_h = _ch6_var_stack_totals(W, mu, cand, ax, scale=scale, positive_only=False)
        score = mean_dev * 50.0 + abs(pos_h - neg_h)
        if score < best_score:
            best, best_score = cand, score
    rng.shuffle(best)
    return best[: n_each * 2]


def _ch6_var_pick_demo_squared(W, mu, axis, *, n=40, seed=115):
    """Balanced +/- subset for positive-only variance piles (same picker as signed)."""
    return _ch6_var_pick_demo(W, mu, axis, n=n, seed=seed)


def _ch6_var_draw_positive_pile(ax, blocks, *, ymax):
    ym = float(max(ymax, 0.06))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, ym)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.9)
        sp.set_edgecolor("#9ca3af")
    ax.axhline(0.0, color="#374151", lw=1.4, zorder=3)
    ax.grid(axis="y", alpha=0.12)
    x0, bar_w = _ch6_cov_pile_geom()
    cum = 0.0
    for h, col, _sign in blocks:
        hh = float(h)
        if hh <= 1e-9:
            continue
        ax.add_patch(
            plt.Rectangle(
                (x0, cum), bar_w, hh,
                facecolor=col, edgecolor="#374151", linewidth=0.9,
                alpha=0.94, zorder=4,
            )
        )
        cum += hh


def _ch6_var_draw_flying_contrib(
    fig, ax3d, ax_pile, *, W, mu, idx, axis, height, color, sign,
    pos_cum, neg_cum, t_fly, ymax, positive_only=False,
):
    smooth = _g("ch3_knob_smoothstep")
    u = float(smooth(float(t_fly)))
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    pt = np.array([W[int(idx), 0], W[int(idx), 1], W[int(idx), 2]], dtype=np.float64)
    sx, sy = _ch6_ax3d_point_to_fig(fig, ax3d, pt)
    x0, bar_w = _ch6_cov_pile_geom()
    h_vis = float(height) * u
    if positive_only:
        y_target = float(pos_cum) + h_vis * 0.5
        ym = float(max(ymax, 0.06))
    else:
        ym = float(max(ymax, 0.06))
        if float(sign) > 0.0:
            y_target = float(pos_cum) + h_vis * 0.5
        else:
            y_target = float(neg_cum) - h_vis * 0.5
    tx, ty = _ch6_ax_data_to_fig(fig, ax_pile, x0 + bar_w * 0.5, y_target)
    if positive_only:
        fw1 = bar_w * ax_pile.get_position().width
        fh1 = (h_vis / ym) * ax_pile.get_position().height
    else:
        fw1 = bar_w * ax_pile.get_position().width
        fh1 = (h_vis / (2.0 * ym)) * ax_pile.get_position().height
    fw0 = fh0 = 0.003
    fw = fw0 + u * (fw1 - fw0)
    fh = fh0 + u * (fh1 - fh0)
    cx = sx + u * (tx - sx)
    cy = sy + u * (ty - sy)
    fig.patches.append(
        plt.Rectangle(
            (cx - fw * 0.5, cy - fh * 0.5), fw, fh,
            transform=fig.transFigure,
            facecolor=color, edgecolor="#374151", linewidth=0.9,
            alpha=0.96, zorder=520,
        )
    )


def _ch6_var_measure_header_content(fig, block, *, style, panel_w_frac: float):
    """Return (content_h_px, n_lines) for an adaptive variance formula box."""
    import handwrite_tutorial as hw

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    preview = hw.block_display_lines(block, style=style)
    body_fs = float(block.get("block_fs", CH6_VARIANCE_MATH_FS))
    min_fs = float(block.get("block_fs_min", 14.0))
    line_dy = hw.block_pt_to_px(fig, block, float(block.get("line_dy_pt", 24.0)))
    panel_w_px = max(float(panel_w_frac) * float(fig.bbox.width), 1.0)
    max_line_px = panel_w_px * float(block.get("mathtext_max_line_frac", 0.94 * CH6_COV_PANEL_BOX_W))
    bold_lhs = bool(block.get("bold_lhs", False))
    if bool(block.get("fit_to_column_width")):
        fp_test = hw.hand_font(body_fs, bold=bold_lhs) if style.enabled else None
        while body_fs > min_fs and preview:
            widest = 0.0
            for raw in preview:
                runs = hw.parse_handwrite_runs(raw)
                widest = max(
                    widest,
                    hw.mixed_line_width_px(
                        renderer, runs, body_fs, style=style, bold=bold_lhs, fp_hand=fp_test,
                    ),
                )
            if widest <= max_line_px:
                break
            body_fs -= 0.5
    lines: list[str] = []
    for raw in preview:
        runs = hw.parse_handwrite_runs(raw)
        w_px = hw.mixed_line_width_px(
            renderer, runs, body_fs, style=style,
            bold=bold_lhs,
            fp_hand=hw.hand_font(body_fs, bold=bold_lhs) if style.enabled else None,
        )
        if w_px <= max_line_px or not bool(block.get("force_wrap", False)):
            lines.append(raw)
        else:
            lines.extend(
                hw.wrap_text_for_width(
                    fig, raw, body_fs, max_line_px / float(fig.bbox.width),
                    bold=False, style=style,
                )
            )
    content_h_px = 0.0
    fp_line = hw.hand_font(body_fs, bold=bold_lhs) if style.enabled else None
    for raw in lines:
        line_h = hw.mixed_line_height_px(
            renderer, hw.parse_handwrite_runs(raw), body_fs, style=style,
            bold=bold_lhs, fp_hand=fp_line,
        )
        content_h_px += line_h
    if len(lines) > 1:
        content_h_px += line_dy * float(len(lines) - 1)
    return float(content_h_px), int(len(lines))


def _ch6_draw_math_formula_red_box(ax_top, *, alpha: float = 0.98):
    """Red rounded box around a formula cell (matches ch6_120)."""
    ax_top.add_patch(
        FancyBboxPatch(
            (CH6_COV_PANEL_BOX_X, CH6_COV_PANEL_BOX_Y),
            CH6_COV_PANEL_BOX_W, CH6_COV_PANEL_BOX_H,
            transform=ax_top.transAxes,
            boxstyle="round,pad=0.012",
            edgecolor=CH6_VARIANCE_RED,
            facecolor="none",
            linewidth=float(CH6_COV_PANEL_BOX_LW),
            alpha=float(alpha),
            zorder=5,
        )
    )


def _ch6_var_box_pad_px(fig, block, hw):
    """Top/bottom inset padding inside the red formula box (asymmetric)."""
    legacy = block.get("box_pad_pt")
    if block.get("box_pad_top_pt") is not None:
        pad_top_pt = float(block["box_pad_top_pt"])
    elif legacy is not None:
        pad_top_pt = float(legacy)
    else:
        pad_top_pt = CH6_VARIANCE_MATH_BOX_PAD_TOP_PT
    if block.get("box_pad_bottom_pt") is not None:
        pad_bottom_pt = float(block["box_pad_bottom_pt"])
    elif legacy is not None:
        pad_bottom_pt = float(legacy)
    else:
        pad_bottom_pt = CH6_VARIANCE_MATH_BOX_PAD_BOTTOM_PT
    return (
        hw.block_pt_to_px(fig, block, pad_top_pt),
        hw.block_pt_to_px(fig, block, pad_bottom_pt),
    )


def _ch6_var_draw_formula_above_pile(
    fig,
    block,
    *,
    style,
    panel_x: float,
    panel_w: float,
    pile_y0: float,
    pile_h: float,
    header_done: bool,
    line_progress,
    gap_frac: float = 0.010,
):
    """Draw variance formula in a red box just above the pile panel (same width)."""
    import handwrite_tutorial as hw

    pad_top_px, pad_bottom_px = _ch6_var_box_pad_px(fig, block, hw)
    content_h_px, _n_lines = _ch6_var_measure_header_content(
        fig, block, style=style, panel_w_frac=panel_w,
    )
    inner_h_px = content_h_px + pad_top_px + pad_bottom_px
    box_h_px = inner_h_px / CH6_COV_PANEL_BOX_H
    fig_h_px = max(float(fig.bbox.height), 1.0)
    box_h_frac = box_h_px / fig_h_px
    box_y = float(pile_y0) + float(pile_h) + float(gap_frac)
    ax_top = fig.add_axes(
        [float(panel_x), box_y, float(panel_w), box_h_frac],
        zorder=515, facecolor="none",
    )
    ax_top.set_axis_off()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_h_px = max(float(ax_top.get_window_extent(renderer).height), 1.0)
    inset_top_px = ax_h_px * CH6_COV_PANEL_BOX_Y
    text_y_inset_pt = (inset_top_px + pad_top_px) * 72.0 / float(fig.dpi)
    if header_done:
        _ch6_draw_math_formula_red_box(ax_top, alpha=0.98)
    hw.draw_block_cell(
        ax_top, block, style=style,
        block_fs=float(block.get("block_fs", CH6_VARIANCE_MATH_FS)),
        label_fs=14.0,
        align=str(block.get("align", "center")),
        text_x_frac=float(block.get("text_x_frac", 0.5)),
        text_y_inset_pt=float(text_y_inset_pt), text_color="#1f2937",
        show_frame=False,
        line_progress=line_progress,
    )


def _ch6_apply_cov_math_left_panel(
    fig,
    ax_data,
    axes_k,
    ax3d,
    *,
    left_fade_u: float,
    header_u: float,
    contrib_landed: list,
    contrib_fly,
    plane_b: float = 0.0,
    contrib_ymax=None,
    header_block=None,
    positive_only_pile: bool = False,
    var_axis: int = 0,
    var_flying: bool = False,
):
    """Fade plot + knobs; pinned Cov formula; flying signed contribution blocks."""
    fade_u = float(np.clip(left_fade_u, 0.0, 1.0))
    header_u = float(np.clip(header_u, 0.0, 1.0))
    if fade_u <= 1e-6 and header_u <= 1e-6 and not contrib_landed and not contrib_fly:
        return

    if fade_u > 0.15:
        _ch6_strip_left_data_ui(ax_data, axes_k)
    _ch6_fade_left_panel_artists(ax_data, axes_k, fade_u)
    rect = _ch6_left_panel_rect(ax_data, axes_k)
    if fade_u > 1e-6:
        fig.patches.append(
            plt.Rectangle(
                (rect[0], rect[1]), rect[2], rect[3],
                transform=fig.transFigure,
                facecolor="#fafafa",
                edgecolor="none",
                alpha=1.0,
                zorder=500,
            )
        )

    import handwrite_tutorial as hw

    composer = _g("CH4_COMPOSER")
    style = composer.handwrite_style()

    pile_h = 0.72 * rect[3]
    pile_y0 = rect[1] + 0.06 * rect[3]
    header_h = max(0.14 * rect[3], rect[1] + rect[3] - pile_y0 - pile_h - 0.02 * rect[3])
    panel_x, panel_w = _ch6_cov_panel_hrect(rect)

    header_block_obj = None
    header_done = False
    line_progress: dict = {}
    if header_u > 1e-6:
        header_block_obj = header_block if header_block is not None else _ch6_cov_math_header_block()
        n_lines = len(hw.block_display_lines(header_block_obj, style=style))
        header_done = float(header_u) >= 1.0 - 1e-9
        if header_done:
            line_progress = {i: 1.0 for i in range(int(n_lines))}
        else:
            line_progress = _ch6_cov_math_line_progress(n_lines, header_u)

        if not var_flying:
            ax_top = fig.add_axes(
                [rect[0], pile_y0 + pile_h + 0.02 * rect[3], rect[2], header_h],
                zorder=510, facecolor="none",
            )
            ax_top.set_axis_off()
            text_y_inset_pt = 8.0
            if str(header_block_obj.get("valign", "")) == "center":
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                ax_h_px = max(float(ax_top.get_window_extent(renderer).height), 1.0)
                body_fs = float(header_block_obj.get("block_fs", CH6_COV_MATH_FS))
                line_dy = hw.block_pt_to_px(
                    fig, header_block_obj, float(header_block_obj.get("line_dy_pt", 24.0)),
                )
                content_h = 0.0
                for raw in hw.block_display_lines(header_block_obj, style=style):
                    content_h += hw.text_line_height_px(
                        renderer, raw, body_fs, style=style,
                    )
                if int(n_lines) > 1:
                    content_h += line_dy * float(int(n_lines) - 1)
                box_h_px = ax_h_px * 0.92
                text_y_inset_pt = max(4.0, 0.5 * (box_h_px - content_h))
            hw.draw_block_cell(
                ax_top, header_block_obj, style=style,
                block_fs=float(header_block_obj.get("block_fs", CH6_COV_MATH_FS)),
                label_fs=14.0,
                align=str(header_block_obj.get("align", "left")),
                text_x_frac=float(header_block_obj.get("text_x_frac", 0.05)),
                text_y_inset_pt=float(text_y_inset_pt), text_color="#1f2937",
                show_frame=False,
                line_progress=line_progress,
            )
            if header_done:
                _ch6_draw_math_formula_red_box(ax_top, alpha=0.98)

    show_pile = (
        header_u > 0.85
        and (contrib_landed or contrib_fly or contrib_ymax is not None)
    )
    show_var_panel = bool(var_flying and header_u > 1e-6)

    if show_var_panel or show_pile:
        ax_pile = fig.add_axes(
            [panel_x, pile_y0, panel_w, pile_h],
            zorder=510, facecolor="none",
        )
        if show_pile:
            ymax = (
                float(contrib_ymax)
                if contrib_ymax is not None
                else (
                    _ch6_var_positive_ymax(contrib_landed, fly=contrib_fly)
                    if positive_only_pile
                    else _ch6_cov_signed_ymax(contrib_landed, fly=contrib_fly)
                )
            )
            if positive_only_pile:
                _ch6_var_draw_positive_pile(ax_pile, contrib_landed, ymax=ymax)
            else:
                _ch6_cov_draw_signed_pile(ax_pile, contrib_landed, ymax=ymax)
            if contrib_fly is not None:
                fidx, fh, fcol, fsign, ft, fpos, fneg, fW, fmu = contrib_fly
                if var_flying:
                    _ch6_var_draw_flying_contrib(
                        fig, ax3d, ax_pile,
                        W=fW, mu=fmu,
                        idx=int(fidx), axis=int(var_axis),
                        height=float(fh), color=str(fcol), sign=float(fsign),
                        pos_cum=float(fpos), neg_cum=float(fneg), t_fly=float(ft),
                        ymax=ymax, positive_only=bool(positive_only_pile),
                    )
                else:
                    _ch6_cov_draw_flying_contrib(
                        fig, ax3d, ax_pile,
                        W=fW, mu=fmu,
                        idx=int(fidx), height=float(fh), color=str(fcol), sign=float(fsign),
                        pos_cum=float(fpos), neg_cum=float(fneg), t_fly=float(ft),
                        ymax=ymax, plane_b=float(plane_b),
                    )
        if show_var_panel and header_block_obj is not None:
            _ch6_var_draw_formula_above_pile(
                fig, header_block_obj, style=style,
                panel_x=panel_x, panel_w=panel_w,
                pile_y0=pile_y0, pile_h=pile_h,
                header_done=header_done, line_progress=line_progress,
                gap_frac=0.012 * float(rect[3]),
            )


def _ch6_cov_pick_demo(W, mu, *, n_each=20, seed=119):
    """Sample demo indices in each w_ST–w_EL quadrant."""
    rng = np.random.default_rng(int(seed))
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    d0 = W[:, 0] - mu[0]
    d1 = W[:, 1] - mu[1]
    out: dict[str, list[int]] = {}
    for key, mask in (
        ("pp", (d0 > 0.0) & (d1 > 0.0)),
        ("nn", (d0 < 0.0) & (d1 < 0.0)),
        ("pn", (d0 > 0.0) & (d1 < 0.0)),
        ("np", (d0 < 0.0) & (d1 > 0.0)),
    ):
        idx = np.flatnonzero(mask)
        if len(idx) > int(n_each):
            idx = rng.choice(idx, size=int(n_each), replace=False)
        out[key] = sorted(int(i) for i in idx)
    return out


def _ch6_cov_warp_cloud(W, mu, mode):
    """Correlation variants in the w_ST–w_EL plane (b fixed at μ)."""
    W = np.asarray(W, dtype=np.float64).copy()
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    W[:, 2] = mu[2]
    if mode == "positive":
        return W
    if mode == "negative":
        W[:, 1] = 2.0 * mu[1] - W[:, 1]
        return W
    if mode == "independent":
        rng = np.random.default_rng(119)
        n = len(W)
        st_s = 0.34 * (CH6_COV_WST_VIEW[1] - CH6_COV_WST_VIEW[0])
        el_s = 0.34 * abs(CH6_COV_WEL_VIEW[1] - CH6_COV_WEL_VIEW[0])
        z = rng.standard_normal((n, 2))
        W[:, 0] = mu[0] + z[:, 0] * st_s
        W[:, 1] = mu[1] + z[:, 1] * el_s
        W[:, 0] = np.clip(W[:, 0], CH6_COV_WST_VIEW[0], CH6_COV_WST_VIEW[1])
        W[:, 1] = np.clip(W[:, 1], CH6_COV_WEL_VIEW[0], CH6_COV_WEL_VIEW[1])
        return W
    return W


def _ch6_cov_interp_cloud(W_a, W_b, u):
    u = float(np.clip(u, 0.0, 1.0))
    return (1.0 - u) * np.asarray(W_a, dtype=np.float64) + u * np.asarray(W_b, dtype=np.float64)


def _ch6_cov_accum_series(W, mu, order, axis_lim):
    """Cumulative covariance contributions and line lengths for ``order`` indices."""
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    span = hi - lo
    running = 0.0
    lengths: list[float] = []
    for idx in order:
        prod = float(W[int(idx), 0] - mu[0]) * float(W[int(idx), 1] - mu[1])
        running += prod
        lengths.append(running)
    peak = max((abs(v) for v in lengths), default=1.0)
    scale = 0.34 * span / max(peak, 1e-12)
    return [v * scale for v in lengths]


def _ch6_118_end_state(pack):
    """End pose of ch6_118 (eigen-arrows, no box, post-orbit azimuth)."""
    end = _ch6_117_end_state(pack)
    return {**end, "azim": float(end["azim"]) + 90.0 + 360.0}


def _ch6_117_end_state(pack):
    """End pose of ch6_117 (original cloud, final azimuth, box + ranges)."""
    info = _ch6_variance_info_from_114_pack(pack)
    W_orig = np.asarray(info["W"], dtype=np.float64)
    axis_lim = info["marker_axis_lim"]
    full_ranges, box_bounds, inner_bounds = _ch6_variance_end_ranges(W_orig)
    targets = _ch6_warp_target_clouds_in_box(W_orig, box_bounds, axis_lim, seed=117)
    init_elev, init_azim = _variance_init_view()
    end_azim = float(init_azim) + 360.0
    n_legs = len((
        targets["original"],
        targets["isotropic"],
        targets["bimodal"],
        targets["elongated"],
        targets["ring"],
        targets["original"],
    )) - 1
    az_final = end_azim + 360.0 * float(n_legs)
    return {
        "info": info,
        "W": np.asarray(targets["original"], dtype=np.float64),
        "mu": np.asarray(info["mu"], dtype=np.float64).reshape(3),
        "axis_lim": axis_lim,
        "full_ranges": full_ranges,
        "box_bounds": box_bounds,
        "inner_bounds": inner_bounds,
        "elev": float(init_elev),
        "azim": float(az_final),
        "line_color": CH6_VARIANCE_POS_COLOR,
    }


def _ch6_fit_cloud_plane(W):
    """Best-fit affine plane for a 3D cloud: center + two in-plane orthonormal axes."""
    W = np.asarray(W, dtype=np.float64)
    mu = np.mean(W, axis=0)
    _, _, vh = np.linalg.svd(W - mu, full_matrices=False)
    e1 = np.asarray(vh[0], dtype=np.float64)
    e2 = np.asarray(vh[1], dtype=np.float64)
    return mu, e1, e2


def _ch6_warp_target_clouds(W_orig, axis_lim, *, seed=89):
    """Synthetic (w_ST, w_EL, b) targets for the ch6_100 warp reel."""
    rng = np.random.default_rng(int(seed))
    W_orig = np.asarray(W_orig, dtype=np.float64)
    n = len(W_orig)
    mu = W_orig.mean(axis=0)
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    span = hi - lo

    iso = mu + rng.normal(0.0, 0.28 * span, size=(n, 3))

    off = np.array([0.38 * span, 0.22 * span, 0.12 * span], dtype=np.float64)
    n0 = n // 2
    bimodal = np.vstack([
        mu + off + rng.normal(0.0, 0.10 * span, (n0, 3)),
        mu - off + rng.normal(0.0, 0.10 * span, (n - n0, 3)),
    ])
    rng.shuffle(bimodal, axis=0)

    elongated = np.column_stack([
        rng.normal(mu[0], 0.42 * span, n),
        rng.normal(mu[1], 0.42 * span, n),
        np.full(n, mu[2]) + rng.normal(0.0, 0.02 * span, n),
    ])

    targets = {"original": _ch6_order_match_cloud(W_orig, W_orig.copy())}
    for key, arr in (
        ("isotropic", iso),
        ("bimodal", bimodal),
        ("elongated", elongated),
    ):
        targets[key] = _ch6_prepare_warp_cloud(W_orig, arr, axis_lim)
    return targets


def _ch6_planar_warp_target_clouds(W_orig, *, seed=110):
    """Synthetic targets constrained to the best-fit plane of ``W_orig``."""
    rng = np.random.default_rng(int(seed))
    W_orig = np.asarray(W_orig, dtype=np.float64)
    n = len(W_orig)
    mu, e1, e2 = _ch6_fit_cloud_plane(W_orig)
    B = np.column_stack([e1, e2])  # 3x2
    uv_orig = (W_orig - mu) @ B

    u_mu, v_mu = float(np.mean(uv_orig[:, 0])), float(np.mean(uv_orig[:, 1]))
    u_span = max(float(np.ptp(uv_orig[:, 0])), 1e-6)
    v_span = max(float(np.ptp(uv_orig[:, 1])), 1e-6)

    iso_uv = np.column_stack([
        rng.normal(u_mu, 0.30 * u_span, n),
        rng.normal(v_mu, 0.30 * v_span, n),
    ])

    du, dv = 0.36 * u_span, 0.20 * v_span
    n0 = n // 2
    bimodal_uv = np.vstack([
        np.column_stack([
            rng.normal(u_mu + du, 0.10 * u_span, n0),
            rng.normal(v_mu + dv, 0.10 * v_span, n0),
        ]),
        np.column_stack([
            rng.normal(u_mu - du, 0.10 * u_span, n - n0),
            rng.normal(v_mu - dv, 0.10 * v_span, n - n0),
        ]),
    ])
    rng.shuffle(bimodal_uv, axis=0)

    elongated_uv = np.column_stack([
        rng.normal(u_mu, 0.46 * u_span, n),
        rng.normal(v_mu, 0.08 * v_span, n),
    ])

    # Keep synthetic distributions within the original cloud footprint in plane.
    u_lo, u_hi = float(np.min(uv_orig[:, 0])), float(np.max(uv_orig[:, 0]))
    v_lo, v_hi = float(np.min(uv_orig[:, 1])), float(np.max(uv_orig[:, 1]))

    def to_xyz(uv):
        uv = np.asarray(uv, dtype=np.float64)
        uv[:, 0] = np.clip(uv[:, 0], u_lo, u_hi)
        uv[:, 1] = np.clip(uv[:, 1], v_lo, v_hi)
        return mu + uv @ B.T

    targets = {"original": _ch6_order_match_cloud(W_orig, W_orig.copy())}
    for key, uv in (
        ("isotropic", iso_uv),
        ("bimodal", bimodal_uv),
        ("elongated", elongated_uv),
    ):
        targets[key] = _ch6_order_match_cloud(W_orig, to_xyz(uv))
    return targets


def _ch6_cloud_frame_from_pack(
    pack,
    *,
    markers,
    axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
    view_azim=None,
    keep_classroom=True,
    ghost_fade_u=1.0,
    marker_density_reveal_u=1.0,
    ghost_density_reveal_u=0.0,
):
    """Duo frame: 2D roster + 3D marker cloud (no density histograms)."""
    last = pack["reel_steps"][-1]
    cs, ce, cy = last["study"], last["exam"], last["y"]
    w = last["w"]
    ghosts = last["ghosts"]
    markers = np.asarray(markers, dtype=np.float64)
    xlim, ylim = pack["xlim"], pack["ylim"]
    pop_s, pop_e, pop_y = pack["pop_s"], pack["pop_e"], pack["pop_y"]
    roster_kw = (
        dict(base_study=cs, base_exam=ce, base_y=cy, show_base=True)
        if keep_classroom
        else dict(base_study=cs[:0], base_exam=ce[:0], base_y=cy[:0], show_base=False)
    )
    return _frame_duo(
        xlim=xlim, ylim=ylim,
        pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
        ghost_ws=ghosts, knob_w=w,
        markers=markers, highlight_w=w,
        view_azim=view_azim,
        ghost_fade_u=ghost_fade_u,
        marker_z_mode="bias", marker_axis_lim=axis_lim, zlabel="b",
        marker_density_reveal_u=float(marker_density_reveal_u),
        ghost_density_reveal_u=float(ghost_density_reveal_u),
        show_legend=False, title_left=None, title_right=None,
        **roster_kw,
    )


def _ch6_estimate_dist_reel_frame_count(
    pack,
    *,
    fast,
    keep_classroom_on_stage,
    density_at_end,
    density_end_compact,
    density_histograms,
    rotate_continuous,
    reel_render_stride=1,
):
    """Budget frames for a full 360° azimuth sweep across the dist-reel clip."""
    has_match = pack["match_s"] is not None
    n_hold_open = CH6_N_HOLD if fast else CH6_N_HOLD * 2
    n_hold_pop = max(1, CH6_N_HOLD // 2) if fast else CH6_N_HOLD
    n_flash = 0 if fast else CH6_N_FLASH
    n_seq_hold = 0 if fast else CH6_N_SEQ_HOLD
    reel_render_stride = max(1, int(reel_render_stride))
    n_reel_steps = len(pack["reel_steps"])
    n_reel_vis = sum(
        1 for i in range(n_reel_steps)
        if i % reel_render_stride == 0 or i == n_reel_steps - 1
    )
    if rotate_continuous:
        n_hold_open = min(n_hold_open, 1)
        n_hold_pop = 0
        n_flash = 0
        n_seq_hold = 0

    n = 1 + n_hold_open
    n += _draft_short(12, 5) + n_hold_pop
    if has_match:
        n += _draft_short(10, 4) + n_hold_pop
    per_step = 1 + n_flash + n_seq_hold
    if not keep_classroom_on_stage:
        per_step += n_seq_hold
    n += n_reel_vis * per_step
    n += CH6_N_HOLD

    if density_at_end:
        n_hold_d = 1 if rotate_continuous else (
            CH6_N_HOLD if density_end_compact else CH6_N_HOLD * 2
        )
        n += _draft_short(16 if density_end_compact else 32, 8)
        n += 1 + n_hold_d
        n += _draft_short(36 if density_end_compact else 72, 18)
        n += 1 + n_hold_d
        if not rotate_continuous:
            n += _draft_short(36 if density_end_compact else 72, 14) + 1
        n += (1 if rotate_continuous else CH6_N_HOLD * 2)
    elif density_histograms:
        n += CH6_N_HOLD * 2
        n += _draft_short(72, 28) + 1 + CH6_N_HOLD * 2
    else:
        n += _draft_short(14, 6) + _draft_short(8, 4)
        n += 1 + CH6_N_HOLD * 2
        n += _draft_short(72, 28) + 1 + CH6_N_HOLD * 2
    return int(n)


def _frame_ch6_dual_reel_2d(
    layout_u,
    right_u,
    *,
    xlim,
    ylim,
    left_base_study,
    left_base_exam,
    left_base_y,
    left_pop_study=None,
    left_pop_exam=None,
    left_pop_y=None,
    left_pop_alpha=0.16,
    left_w=None,
    left_ghosts=None,
    right_base_study=None,
    right_base_exam=None,
    right_base_y=None,
    right_pop_study=None,
    right_pop_exam=None,
    right_pop_y=None,
    right_pop_alpha=0.16,
    right_w=None,
    right_ghosts=None,
    show_right=False,
    show_legend=False,
):
    """Morph wide → left duo slot; optional right 2D reel panel (no knobs)."""
    u = float(np.clip(float(layout_u), 0.0, 1.0))
    ru = float(np.clip(float(right_u), 0.0, 1.0))
    su = float(_g("ch3_knob_smoothstep")(u))
    sr = float(_g("ch3_knob_smoothstep")(ru)) if show_right else 0.0
    wide_data, left_tgt, right_tgt = _ch6_dual_2d_layout()
    left_r = _g("_ch4_02b_lerp_rect")(su, wide_data, left_tgt)

    fig = plt.figure(figsize=_g("CH4_DUO_FIGSIZE"))
    fig.patch.set_facecolor("white")
    ax_l = fig.add_axes(left_r)
    _ch6_draw_2d_reel_panel(
        ax_l,
        xlim=xlim, ylim=ylim,
        base_study=left_base_study, base_exam=left_base_exam, base_y=left_base_y,
        pop_study=left_pop_study, pop_exam=left_pop_exam, pop_y=left_pop_y,
        pop_alpha=left_pop_alpha,
        w_live=left_w, ghost_ws=left_ghosts,
        show_legend=show_legend,
    )
    if show_right and sr > 1e-3 and right_base_study is not None:
        ax_r = fig.add_axes(right_tgt)
        ax_r.patch.set_alpha(sr)
        for spine in ax_r.spines.values():
            spine.set_alpha(sr)
        _ch6_draw_2d_reel_panel(
            ax_r,
            xlim=xlim, ylim=ylim,
            base_study=right_base_study, base_exam=right_base_exam, base_y=right_base_y,
            pop_study=right_pop_study, pop_exam=right_pop_exam, pop_y=right_pop_y,
            pop_alpha=float(right_pop_alpha) * sr,
            w_live=right_w, ghost_ws=right_ghosts, ghost_fade_u=sr,
            show_legend=show_legend,
        )
    return _fig_to_plot_wide(fig)


def _frame_ch6_wide_2d(
    *,
    xlim,
    ylim,
    base_study,
    base_exam,
    base_y,
    show_base=True,
    pop_study=None,
    pop_exam=None,
    pop_y=None,
    pop_alpha=0.16,
    w_live=None,
    ghost_ws=None,
    ghost_fade_u=1.0,
    knob_w=None,
    show_legend=False,
):
    """Full-bleed wide 2D panel (ch4_02b layout) — no 3D, no class Gaussians."""
    wide_data, wide_knobs = _g("_ch4_02b_wide_layout")()
    fig = plt.figure(figsize=_g("CH4_DUO_FIGSIZE"))
    fig.patch.set_facecolor("white")
    ax_data = fig.add_axes(wide_data)
    _ch6_draw_2d_reel_panel(
        ax_data,
        xlim=xlim, ylim=ylim,
        base_study=base_study, base_exam=base_exam, base_y=base_y,
        show_base=show_base,
        pop_study=pop_study, pop_exam=pop_exam, pop_y=pop_y, pop_alpha=pop_alpha,
        w_live=w_live, ghost_ws=ghost_ws, ghost_fade_u=ghost_fade_u,
        show_legend=show_legend,
    )
    kw = knob_w if knob_w is not None else (w_live if w_live is not None else CH6_W_TRUE)
    ws, we, bb = float(kw[0]), float(kw[1]), float(kw[2])
    _ch6_place_knobs_wide(fig, ax_data, wide_data, wide_knobs, ws, we, bb, grow_u=1.0)
    return _fig_to_plot_wide(fig)


def _frame_ch6_duo_cloud_layout(
    *,
    layout_u=0.0,
    cloud_u=1.0,
    knob_u=1.0,
    markers,
    highlight_w,
    view_azim=None,
    marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
    marker_s=28,
    highlight_marker_s=70,
    xlim,
    ylim,
    base_study,
    base_exam,
    base_y,
    pop_study=None,
    pop_exam=None,
    pop_y=None,
    pop_alpha=0.08,
    w_live=None,
    knob_w=None,
    ghost_ws=None,
    ghost_fade_u=0.0,
    marker_density_reveal_u=0.0,
    ghost_density_reveal_u=0.0,
    show_legend=False,
):
    """``cloud_u`` / ``knob_u`` fade at duo size; ``layout_u`` resizes 2D afterward."""
    su = float(_g("ch3_knob_smoothstep")(float(np.clip(float(layout_u), 0.0, 1.0))))
    cu = float(np.clip(float(cloud_u), 0.0, 1.0))
    ku = float(np.clip(float(knob_u), 0.0, 1.0))
    wide_data, _ = _g("_ch4_02b_wide_layout")()
    duo_data, duo_knobs, duo_3d = _ch6_weight3d_duo_layout()
    data_r = _g("_ch4_02b_lerp_rect")(su, duo_data, wide_data)

    fig = plt.figure(figsize=_g("CH4_DUO_FIGSIZE"))
    fig.patch.set_facecolor("white")
    ax_data = fig.add_axes(data_r)
    _ch6_draw_2d_reel_panel(
        ax_data,
        xlim=xlim, ylim=ylim,
        base_study=base_study, base_exam=base_exam, base_y=base_y,
        pop_study=pop_study, pop_exam=pop_exam, pop_y=pop_y,
        pop_alpha=float(pop_alpha) * (1.0 - 0.85 * su),
        w_live=w_live, ghost_ws=ghost_ws, ghost_fade_u=ghost_fade_u,
        markers=markers,
        ghost_density_reveal_u=ghost_density_reveal_u,
        marker_axis_lim=marker_axis_lim,
        show_legend=show_legend,
    )

    kw = knob_w if knob_w is not None else (w_live if w_live is not None else CH6_W_TRUE)
    ws, we, bb = float(kw[0]), float(kw[1]), float(kw[2])
    if ku > 1e-4 and su < 1e-4:
        _ch6_place_knobs_duo_fade(fig, ax_data, ws, we, bb, fade_u=ku)

    if cu > 1e-3 and markers is not None:
        ax3d = fig.add_axes(duo_3d, projection="3d")
        lo, hi = float(marker_axis_lim[0]), float(marker_axis_lim[1])
        z_lim = (lo, hi)
        _style_ax3d(ax3d, z_lim=z_lim, zlabel="b", xy_lim=(lo, hi), azim=view_azim)
        ax3d.patch.set_alpha(cu)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.pane.fill = False
            axis.pane.set_alpha(cu * 0.35)
        Wm = np.asarray(markers, dtype=np.float64)
        mcols = _ch6_marker_density_colors(
            Wm,
            reveal_u=marker_density_reveal_u,
            z_lim=marker_axis_lim,
        ) if float(marker_density_reveal_u) > 1e-6 else None
        _draw_markers(
            ax3d, Wm, s=marker_s, alpha=0.85 * cu, colors=mcols,
            z=Wm[:, 2] if Wm.ndim == 2 and Wm.shape[1] >= 3 else None,
        )
        if highlight_w is not None:
            hw = np.asarray(highlight_w, dtype=np.float64).reshape(1, 3)
            _draw_markers(
                ax3d, hw, color=CH6_LINE_COLOR, s=highlight_marker_s,
                alpha=cu, z=hw[:, 2],
            )
    return _fig_to_plot_wide(fig)


def _frame_ch6_duo_cloud_to_fullscreen(
    layout_u,
    *,
    cloud_u=1.0,
    knob_u=None,
    markers,
    highlight_w,
    view_azim=None,
    marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
    marker_s=28,
    highlight_marker_s=70,
    xlim,
    ylim,
    base_study,
    base_exam,
    base_y,
    pop_study=None,
    pop_exam=None,
    pop_y=None,
    pop_alpha=0.08,
    w_live=None,
    knob_w=None,
    ghost_ws=None,
    ghost_fade_u=0.0,
    show_legend=False,
):
    """Legacy wrapper — use ``_frame_ch6_duo_cloud_layout`` for staged fades."""
    ku = float(knob_u) if knob_u is not None else max(0.0, 1.0 - float(layout_u))
    return _frame_ch6_duo_cloud_layout(
        layout_u=layout_u,
        cloud_u=cloud_u,
        knob_u=ku,
        markers=markers,
        highlight_w=highlight_w,
        view_azim=view_azim,
        marker_axis_lim=marker_axis_lim,
        marker_s=marker_s,
        highlight_marker_s=highlight_marker_s,
        xlim=xlim, ylim=ylim,
        base_study=base_study, base_exam=base_exam, base_y=base_y,
        pop_study=pop_study, pop_exam=pop_exam, pop_y=pop_y,
        pop_alpha=pop_alpha,
        w_live=w_live, knob_w=knob_w,
        ghost_ws=ghost_ws, ghost_fade_u=ghost_fade_u,
        show_legend=show_legend,
    )


def _frame_ch6_fullscreen_2d_grad(
    *,
    xlim,
    ylim,
    base_study,
    base_exam,
    base_y,
    show_base=True,
    w_live=None,
    ghost_ws=None,
    ghost_fade_u=1.0,
    mu_threshold_2d=None,
    classroom_threshold_2d=None,
    show_point_grads_away_from_line=False,
    point_grad_highlight=None,
    grad_span_frac=0.22,
    grad_compact=False,
    show_legend=False,
):
    """Full-screen 2D with optional per-point ∇NLL arrows."""
    wide_data, _ = _g("_ch4_02b_wide_layout")()
    fig = plt.figure(figsize=_g("CH4_DUO_FIGSIZE"))
    fig.patch.set_facecolor("white")
    ax_data = fig.add_axes(wide_data)
    _ch6_draw_2d_reel_panel(
        ax_data,
        xlim=xlim, ylim=ylim,
        base_study=base_study, base_exam=base_exam, base_y=base_y,
        show_base=show_base,
        w_live=None, ghost_ws=ghost_ws, ghost_fade_u=ghost_fade_u,
        show_legend=show_legend,
    )
    if mu_threshold_2d is not None:
        _plot_threshold(
            ax_data, mu_threshold_2d, xlim, ylim,
            color=CH6_VARIANCE_RED, lw=3.6, alpha=1.0, zorder=5,
        )
    if classroom_threshold_2d is not None:
        _plot_threshold(
            ax_data, classroom_threshold_2d, xlim, ylim,
            color="#111111", lw=3.2, ls="--", alpha=1.0, zorder=6,
        )
    elif w_live is not None:
        _plot_threshold(
            ax_data, w_live, xlim, ylim,
            color=CH6_LINE_COLOR, lw=2.5, alpha=1.0, zorder=6,
        )
    if (
        show_point_grads_away_from_line
        and w_live is not None
        and base_study is not None
        and len(base_study) > 0
    ):
        G = ch6_point_nll_grad_contrib(w_live, base_study, base_exam, base_y)
        _draw_point_grad_quivers_away_from_line_2d(
            ax_data, base_study, base_exam, G, w_live,
            highlight_idx=point_grad_highlight,
            span_frac=grad_span_frac,
            compact=grad_compact,
        )
    return _fig_to_plot_wide(fig)


def _frame_ch6_wide_to_duo_empty3d(
    layout_u,
    *,
    xlim,
    ylim,
    base_study,
    base_exam,
    base_y,
    show_base=True,
    pop_study=None,
    pop_exam=None,
    pop_y=None,
    pop_alpha=0.16,
    w_live=None,
    ghost_ws=None,
    ghost_fade_u=1.0,
    knob_w=None,
    show_3d_u=0.0,
    show_legend=False,
):
    """Morph wide 2D → ch4 duo (2D left + knobs + empty 3D Belief axes)."""
    u = float(np.clip(float(layout_u), 0.0, 1.0))
    su = float(_g("ch3_knob_smoothstep")(u))
    s3 = float(np.clip(float(show_3d_u), 0.0, 1.0))
    s3 = float(_g("ch3_knob_smoothstep")(s3))
    wide_data, wide_knobs = _g("_ch4_02b_wide_layout")()
    duo_data, duo_knobs, duo_3d = _ch6_weight3d_duo_layout()
    data_r = _g("_ch4_02b_lerp_rect")(su, wide_data, duo_data)
    knob_rs = tuple(_g("_ch4_02b_lerp_rect")(su, wide_knobs[i], duo_knobs[i]) for i in range(3))

    fig = plt.figure(figsize=_g("CH4_DUO_FIGSIZE"))
    fig.patch.set_facecolor("white")
    ax_data = fig.add_axes(data_r)
    _ch6_draw_2d_reel_panel(
        ax_data,
        xlim=xlim, ylim=ylim,
        base_study=base_study, base_exam=base_exam, base_y=base_y,
        show_base=show_base,
        pop_study=pop_study, pop_exam=pop_exam, pop_y=pop_y, pop_alpha=pop_alpha,
        w_live=w_live, ghost_ws=ghost_ws, ghost_fade_u=ghost_fade_u,
        show_legend=show_legend,
    )
    kw = knob_w if knob_w is not None else (w_live if w_live is not None else CH6_W_TRUE)
    ws, we, bb = float(kw[0]), float(kw[1]), float(kw[2])
    if su > 1e-4:
        _ch6_place_knobs_wide(fig, ax_data, data_r, knob_rs, ws, we, bb, grow_u=su)
    if s3 > 1e-3:
        ax3d = fig.add_axes(duo_3d, projection="3d")
        _style_ax3d(ax3d, z_lim=(0.0, CH6_SURFACE_Z_HI), zlabel="Likelihood")
        ax3d.patch.set_alpha(s3)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.pane.fill = False
            axis.pane.set_alpha(s3 * 0.35)
    return _fig_to_plot_wide(fig)


def _frame_ch6_split_to_duo_empty3d(
    layout_u,
    *,
    right_fade_u=1.0,
    xlim,
    ylim,
    base_study,
    base_exam,
    base_y,
    show_base=True,
    w_live=None,
    ghost_ws=None,
    ghost_fade_u=1.0,
    knob_w=None,
    show_3d_u=0.0,
    right_base_study=None,
    right_base_exam=None,
    right_base_y=None,
    right_w=None,
    right_ghosts=None,
    show_legend=False,
):
    """Morph ch6_98 split-left 2D → ch6_68 duo (knobs + empty 3D); fade right panel out."""
    u = float(np.clip(float(layout_u), 0.0, 1.0))
    su = float(_g("ch3_knob_smoothstep")(u))
    sr = float(_g("ch3_knob_smoothstep")(float(np.clip(float(right_fade_u), 0.0, 1.0))))
    s3 = float(_g("ch3_knob_smoothstep")(float(np.clip(float(show_3d_u), 0.0, 1.0))))
    _, split_left, split_right = _ch6_dual_2d_layout()
    duo_data, duo_knobs, duo_3d = _ch6_weight3d_duo_layout()
    data_r = _g("_ch4_02b_lerp_rect")(su, split_left, duo_data)

    fig = plt.figure(figsize=_g("CH4_DUO_FIGSIZE"))
    fig.patch.set_facecolor("white")
    ax_data = fig.add_axes(data_r)
    _ch6_draw_2d_reel_panel(
        ax_data,
        xlim=xlim, ylim=ylim,
        base_study=base_study, base_exam=base_exam, base_y=base_y,
        show_base=show_base,
        w_live=w_live, ghost_ws=ghost_ws, ghost_fade_u=ghost_fade_u,
        show_legend=show_legend,
    )
    if (
        sr > 1e-3
        and right_base_study is not None
        and len(np.asarray(right_base_study)) > 0
    ):
        ax_r = fig.add_axes(split_right)
        ax_r.patch.set_alpha(sr)
        for spine in ax_r.spines.values():
            spine.set_alpha(sr)
        _ch6_draw_2d_reel_panel(
            ax_r,
            xlim=xlim, ylim=ylim,
            base_study=right_base_study, base_exam=right_base_exam, base_y=right_base_y,
            w_live=right_w, ghost_ws=right_ghosts, ghost_fade_u=sr,
            show_legend=show_legend,
        )
    kw = knob_w if knob_w is not None else (w_live if w_live is not None else CH6_W_TRUE)
    ws, we, bb = float(kw[0]), float(kw[1]), float(kw[2])
    if su > 1e-4:
        _ch6_place_knobs_wide(fig, ax_data, data_r, duo_knobs, ws, we, bb, grow_u=su)
    if s3 > 1e-3:
        ax3d = fig.add_axes(duo_3d, projection="3d")
        lo, hi = (
            float(CH6_POPULATION_PARAM_AXIS_LIM[0]),
            float(CH6_POPULATION_PARAM_AXIS_LIM[1]),
        )
        _style_ax3d(ax3d, z_lim=(lo, hi), zlabel="b", xy_lim=(lo, hi))
        ax3d.patch.set_alpha(s3)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.pane.fill = False
            axis.pane.set_alpha(s3 * 0.35)
    return _fig_to_plot_wide(fig)


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
    data_title=None,
    zlabel="Likelihood",
    p_annotate=False,
    show_legend=True,
    xlabel=None,
    show_point_grads=False,
    show_point_grads_toward_line=False,
    show_point_grads_away_from_line=False,
    line_to_point_arrow_highlight=None,
    point_grad_highlight=None,
    ghost_fade_u=1.0,
    marker_z_mode="floor",
    marker_axis_lim=None,
    view_azim=None,
    camera_zoom=1.0,
    panel_zoom=1.0,
    view_w_st_lim=None,
    view_w_el_lim=None,
    classroom_threshold_2d=None,
    classroom_threshold_2d_u=0.0,
    classroom_threshold_2d_lw=3.2,
    classroom_threshold_2d_color="#111111",
    classroom_threshold_2d_ls="--",
    surface_markers=None,
    surface_marker_s=130,
    ellipsoid_loops=None,
    ellipsoid_alpha=0.85,
    ellipsoid_color=CH6_COV_COLOR,
    ellipsoid_lw=1.6,
    ellipsoid_gradient=False,
    sampling_ellipsoid=None,
    sampling_ellipsoid_reveal_u=1.0,
    show_mean_x=False,
    mean_x=None,
    mean_x_fontsize=30,
    minimal_ui=False,
    marker_s=28,
    highlight_marker_s=70,
    marker_alpha_scale=1.0,
    density_W=None,
    density_axis_lim=None,
    density_reveal_u=1.0,
    density_projection_u=0.0,
    density_bins=24,
    density_bar_pad=None,
    marker_density_reveal_u=0.0,
    ghost_density_reveal_u=0.0,
    show_variance_box=False,
    variance_box_bounds=None,
    gaussian_surface_plane=None,
    gaussian_surface_u=0.0,
    gaussian_surface_grid_n=72,
    gaussian_callout=None,
    gaussian_mu_sigma_u=0.0,
    mu_label_only_u=0.0,
    ch6_marginal_layers=None,
    ch6_marginal_finale_u=0.0,
    ch6_marginal_global_alpha=1.0,
    mu_threshold_2d=None,
    mu_threshold_2d_u=0.0,
    mu_threshold_2d_lw=3.4,
    mu_threshold_2d_color=CH6_VARIANCE_RED,
    gaussian_voxel=None,
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
    g_alpha = _ghost_line_alpha(n_g) * float(np.clip(ghost_fade_u, 0.0, 1.0))
    g_alpha = max(g_alpha, 0.07 * float(np.clip(ghost_fade_u, 0.0, 1.0)))
    ghost_cols = _ch6_ghost_density_colors(
        ghost_ws, markers,
        reveal_u=ghost_density_reveal_u,
        z_lim=marker_axis_lim,
    ) if ghost_ws and float(ghost_density_reveal_u) > 1e-6 else None
    if ghost_ws:
        for gi, wg in enumerate(ghost_ws):
            gcol = (
                ghost_cols[gi]
                if ghost_cols is not None and gi < len(ghost_cols)
                else CH6_GHOST_COLOR
            )
            _plot_threshold(
                ax_data, wg, xlim, ylim,
                color=gcol, lw=1.2, alpha=g_alpha, zorder=3,
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
    if mu_threshold_2d is not None and float(mu_threshold_2d_u) > 1e-6:
        _plot_threshold(
            ax_data, mu_threshold_2d, xlim, ylim,
            color=str(mu_threshold_2d_color), lw=float(mu_threshold_2d_lw),
            alpha=float(np.clip(mu_threshold_2d_u, 0.0, 1.0)), zorder=5,
        )
    if classroom_threshold_2d is not None and float(classroom_threshold_2d_u) > 1e-6:
        _plot_threshold(
            ax_data, classroom_threshold_2d, xlim, ylim,
            color=str(classroom_threshold_2d_color),
            lw=float(classroom_threshold_2d_lw),
            ls=str(classroom_threshold_2d_ls),
            alpha=float(np.clip(classroom_threshold_2d_u, 0.0, 1.0)),
            zorder=6,
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
    if (
        show_point_grads_away_from_line
        and w_live is not None
        and base_study is not None
        and len(base_study) > 0
    ):
        G = ch6_point_nll_grad_contrib(w_live, base_study, base_exam, base_y)
        _draw_point_grad_quivers_away_from_line_2d(
            ax_data, base_study, base_exam, G, w_live,
            highlight_idx=point_grad_highlight,
        )
    if (
        show_point_grads_toward_line
        and w_live is not None
        and base_study is not None
        and len(base_study) > 0
    ):
        _draw_point_arrows_toward_line_2d(
            ax_data, base_study, base_exam, w_live,
            highlight_idx=line_to_point_arrow_highlight,
        )
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
    if data_title:
        ax_data.set_title(
            data_title,
            fontsize=int(_g("AXIS_LABEL_SIZE") * 1.25),
            fontweight="bold",
            pad=16,
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
    x_lim_3d = None
    y_lim_3d = None
    if view_w_st_lim is not None:
        x_lim_3d = (float(view_w_st_lim[0]), float(view_w_st_lim[1]))
    if view_w_el_lim is not None:
        y_lim_3d = (float(view_w_el_lim[0]), float(view_w_el_lim[1]))
    if (
        surface is not None
        and str(surface_palette) == "nll"
        and surface.get("z_lim") is not None
    ):
        zlo, zhi = (float(surface["z_lim"][0]), float(surface["z_lim"][1]))
        zpad = 0.10 * max(zhi - zlo, 1e-6)
        z_lim = (zlo - zpad, zhi + zpad)
    if marker_z_mode == "bias" and surface is None and stems is None:
        if marker_axis_lim is not None:
            lo, hi = float(marker_axis_lim[0]), float(marker_axis_lim[1])
            z_lim = (lo, hi)
            xy_lim = (lo, hi)
        else:
            z_lim = _zlim_from_bias_points(markers=markers, highlight_w=highlight_w)
    _style_ax3d(
        ax3d, z_lim=z_lim, zlabel=zlabel, xy_lim=xy_lim,
        x_lim=x_lim_3d, y_lim=y_lim_3d, azim=view_azim,
        camera_zoom=float(camera_zoom),
        minimal_ui=minimal_ui,
    )
    pz = float(max(panel_zoom, 1e-3))
    if abs(pz - 1.0) > 1e-6:
        bb = ax3d.get_position()
        cx = bb.x0 + 0.5 * bb.width
        cy = bb.y0 + 0.5 * bb.height
        nw = bb.width * pz
        nh = bb.height * pz
        ax3d.set_position([cx - 0.5 * nw, cy - 0.5 * nh, nw, nh])

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
    if (
        gaussian_surface_plane is not None
        and density_W is not None
        and density_axis_lim is not None
    ):
        spec = str(gaussian_surface_plane)
        plane_map = {name: (pi, pj, hk) for name, pi, pj, hk in CH6_DENSITY_PROJ_SPECS}
        if spec in plane_map:
            pi, pj, hk = plane_map[spec]
            _draw_inward_gaussian_surface(
                ax3d, density_W,
                plane_i=pi, plane_j=pj, height_k=hk,
                axis_lim=density_axis_lim,
                side=_density_hist_side(spec, density_W, hk, density_axis_lim),
                reveal_u=float(gaussian_surface_u),
                grid_n=int(gaussian_surface_grid_n),
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
        marker_alpha_scale = float(np.clip(float(marker_alpha_scale), 0.0, 1.0))
        mcols = _ch6_marker_density_colors(
            markers,
            reveal_u=marker_density_reveal_u,
            z_lim=marker_axis_lim,
        ) if float(marker_density_reveal_u) > 1e-6 else None
        if mcols is not None and marker_alpha_scale < 1.0:
            mcols = np.asarray(mcols, dtype=np.float64).copy()
            mcols[:, 3] *= marker_alpha_scale
        if marker_z_mode == "bias":
            _draw_markers(
                ax3d, markers, s=marker_s, colors=mcols,
                z=np.asarray(markers, dtype=np.float64)[:, 2],
                alpha=marker_alpha_scale,
            )
        elif marker_z_mode == "surface" and surface is not None:
            _draw_markers(
                ax3d, markers, s=marker_s, colors=mcols,
                z=_marker_z_on_surface(surface, markers),
                alpha=marker_alpha_scale,
            )
        else:
            _draw_markers(ax3d, markers, s=marker_s, colors=mcols, alpha=marker_alpha_scale)
    if highlight_w is not None:
        hw = np.asarray(highlight_w, dtype=np.float64).reshape(1, 3)
        hcol = CH6_LINE_COLOR if highlight_color is None else str(highlight_color)
        if marker_z_mode == "bias":
            _draw_markers(
                ax3d, hw, color=hcol, s=highlight_marker_s, alpha=marker_alpha_scale,
                z=[float(hw[0, 2])],
            )
        elif marker_z_mode == "surface" and surface is not None:
            _draw_markers(
                ax3d, hw, color=hcol, s=highlight_marker_s, alpha=marker_alpha_scale,
                z=_marker_z_on_surface(surface, hw),
            )
        else:
            _draw_markers(
                ax3d, hw, color=hcol, s=highlight_marker_s, alpha=marker_alpha_scale, z=[0.05],
            )
    if surface_markers and surface is not None and marker_z_mode == "surface":
        for spec in surface_markers:
            pt = np.asarray(spec["point"], dtype=np.float64).reshape(1, 3)
            col = str(spec.get("color", CH6_LINE_COLOR))
            ms = float(spec.get("s", surface_marker_s))
            _draw_markers(
                ax3d, pt, color=col, s=ms, alpha=marker_alpha_scale,
                z=_marker_z_on_surface(surface, pt),
                edgecolors="white",
            )
    if show_variance_box and variance_box_bounds is not None:
        _variance_draw_box(
            ax3d,
            variance_box_bounds,
            edge_alpha=0.9,
            fill_alpha=0.07,
            )

    if ellipsoid_loops:
        _draw_gaussian_ellipsoids(
            ax3d,
            ellipsoid_loops,
            color=str(ellipsoid_color),
            alpha=float(ellipsoid_alpha),
            lw=float(ellipsoid_lw),
            gradient=bool(ellipsoid_gradient),
        )
    if ch6_marginal_layers and markers is not None and len(markers) > 0:
        stats = ch6_param_stats(markers)
        lim = marker_axis_lim if marker_axis_lim is not None else CH6_POPULATION_PARAM_AXIS_LIM
        _draw_ch6_114_marginal_layers(
            ax3d, markers, stats, ch6_marginal_layers,
            axis_lim=lim, marker_s=max(22.0, float(marker_s) * 1.15),
            global_alpha=float(ch6_marginal_global_alpha),
        )
        if float(ch6_marginal_finale_u) > 1e-6:
            _draw_ch6_114_finale_bridges(
                ax3d, stats["mean"],
                axis_lim=lim,
                reveal_u=float(ch6_marginal_finale_u),
                global_alpha=float(ch6_marginal_global_alpha),
            )
    if gaussian_voxel is not None:
        gv = dict(gaussian_voxel)
        _draw_gaussian_voxel_ellipsoid(
            ax3d,
            gv["mean"],
            gv["cov"],
            axis_lim=marker_axis_lim if marker_axis_lim is not None else CH6_POPULATION_PARAM_AXIS_LIM,
            mass=float(gv.get("mass", 0.95)),
            reveal_u=float(gv.get("reveal_u", 1.0)),
            n_cells=int(gv.get("n_cells", 20)),
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
                floor_connector=bool(weight_grad_floor_only),
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
    if markers is not None and len(markers) > 0:
        cloud_mu = np.mean(np.asarray(markers, dtype=np.float64), axis=0)
        if float(mu_label_only_u) > 1e-6:
            _draw_mu_label_only(ax3d, cloud_mu, reveal_u=float(mu_label_only_u))
        elif gaussian_mu_sigma_u > 1e-6:
            _draw_mu_sigma_labels(
                ax3d,
                cloud_mu,
                reveal_u=float(gaussian_mu_sigma_u),
            )
    if gaussian_callout is not None:
        gc = dict(gaussian_callout)
        pt = gc.get("point")
        if pt is not None:
            _draw_gaussian_callout(
                fig, ax3d, pt,
                label=str(gc.get("label", "gaussian")),
                color=str(gc.get("color", CH6_VARIANCE_RED)),
            )

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
    **pop_kw,
):
    """Shared population reel: return classroom landings and opening state."""
    return ch6_population_param_cloud_pack(
        n_class, seed=seed, seed_from_key=seed_from_key, n_reel=n_reel, **pop_kw,
    )


def _build_population_classroom_reel_param_cloud(
    clip_id, *, n_class, seed=20, seed_from_key=None, n_reel=None, fast=False,
    marker_axis_lim=None, keep_classroom_on_stage=False,
    wide_2d_only=False,
    **pop_kw,
):
    """Like ch6_14-17 but right panel is a 3D parameter cloud: (w_ST, w_EL, b).

    When ``wide_2d_only`` is True, render on the ch4 wide 2D canvas with no 3D
  panel and no class Gaussian overlays.
    """
    pack = _population_param_cloud_landings(
        n_class=n_class, seed=seed, seed_from_key=seed_from_key, n_reel=n_reel, **pop_kw,
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

    def _done(img):
        return _finish(img, clip_id)

    def _panel_frame(**extra):
        if wide_2d_only:
            extra.pop("show_ellipses", None)
            extra.pop("markers", None)
            extra.pop("highlight_w", None)
            extra.pop("knob_w", None)
            extra.setdefault("show_legend", False)
            return _frame_ch6_fullscreen_2d(xlim=xlim, ylim=ylim, **extra)
        return _frame_duo(**cloud_kw, **duo_kw, **extra)

    img = _panel_frame(
        base_study=open_s, base_exam=open_e, base_y=open_y,
        w_live=w_open, knob_w=w_open, show_ellipses=not wide_2d_only,
        markers=np.asarray([w_open]) if not wide_2d_only else None,
        highlight_w=w_open if not wide_2d_only else None,
    )
    frames.append(_done(img))
    frames.extend(_hold(frames[-1], n_hold_open))

    for u in np.linspace(0.0, 1.0, _draft_short(12, 5)):
        img = _panel_frame(
            base_study=open_s, base_exam=open_e, base_y=open_y,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
            pop_alpha=0.04 + 0.14 * float(u),
            w_live=w_open, knob_w=w_open,
            show_ellipses=(not wide_2d_only and u < 0.35),
            markers=np.asarray([w_open]) if not wide_2d_only else None,
            highlight_w=w_open if not wide_2d_only else None,
        )
        frames.append(_done(img))
    frames.extend(_hold(frames[-1], n_hold_pop))

    if match_s is not None:
        for u in np.linspace(0.0, 1.0, _draft_short(10, 4)):
            uu = float(u)
            bs = (1.0 - uu) * open_s + uu * match_s
            be = (1.0 - uu) * open_e + uu * pack["match_e"]
            ww = (1.0 - uu) * w_open + uu * w_seed
            img = _panel_frame(
                base_study=bs, base_exam=be, base_y=open_y,
                pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
                w_live=ww, knob_w=ww,
                markers=np.asarray([ww]) if not wide_2d_only else None,
                highlight_w=ww if not wide_2d_only else None,
            )
            frames.append(_done(img))
        frames.extend(_hold(frames[-1], n_hold_pop))

    for step in pack["reel_steps"]:
        cs, ce, cy = step["study"], step["exam"], step["y"]
        w = step["w"]
        markers = step["markers"]
        ghosts = step["ghosts"]
        img = _panel_frame(
            base_study=cs, base_exam=ce, base_y=cy,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
            w_live=w, ghost_ws=ghosts, knob_w=w,
            markers=markers if not wide_2d_only else None,
            highlight_w=w if not wide_2d_only else None,
        )
        frames.append(_done(img))
        frames.extend(_hold(frames[-1], n_flash))
        if not keep_classroom_on_stage:
            img = _panel_frame(
                base_study=cs[:0], base_exam=ce[:0], base_y=cy[:0],
                show_base=False,
                pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
                ghost_ws=ghosts + [w], knob_w=w,
                markers=markers if not wide_2d_only else None,
                highlight_w=w if not wide_2d_only else None,
            )
            frames.append(_done(img))
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
    density_end_compact=False,
    rotate_continuous=False,
    rotate_reel_only=False,
    rotate_reel_start_index=1,
    reel_render_stride=1,
):
    """Point-cloud reel; ellipsoids + spin, live density, or end-of-reel density reveal."""
    return list(_population_reel_param_cloud_dist_frames(
        clip_id,
        n_class=n_class,
        seed=seed,
        seed_from_key=seed_from_key,
        n_reel=n_reel,
        fast=fast,
        marker_axis_lim=marker_axis_lim,
        keep_classroom_on_stage=keep_classroom_on_stage,
        density_histograms=density_histograms,
        density_at_end=density_at_end,
        density_show_mean_x=density_show_mean_x,
        density_bins=density_bins,
        density_bar_pad=density_bar_pad,
        density_end_compact=density_end_compact,
        rotate_continuous=rotate_continuous,
        rotate_reel_only=rotate_reel_only,
        rotate_reel_start_index=rotate_reel_start_index,
        reel_render_stride=reel_render_stride,
    ))


def _population_reel_param_cloud_dist_frames(
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
    density_end_compact=False,
    rotate_continuous=False,
    rotate_reel_only=False,
    rotate_reel_start_index=1,
    reel_render_stride=1,
):
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
    last_frame = None
    reel_render_stride = max(1, int(reel_render_stride))
    duo_kw = dict(show_legend=False, title_left=None, title_right=None)
    n_hold_open = CH6_N_HOLD if fast else CH6_N_HOLD * 2
    n_hold_pop = max(1, CH6_N_HOLD // 2) if fast else CH6_N_HOLD
    n_flash = 0 if fast else CH6_N_FLASH
    n_seq_hold = 0 if fast else CH6_N_SEQ_HOLD
    if rotate_continuous and not rotate_reel_only:
        n_hold_open = min(n_hold_open, 1)
        n_hold_pop = 0
        n_flash = 0
        n_seq_hold = 0
    cloud_kw = dict(marker_z_mode="bias", marker_axis_lim=marker_axis_lim, zlabel="b")
    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))
    rot_total = 0
    rot_i = 0
    n_steps = len(pack["reel_steps"])
    rotate_reel_start_index = int(max(0, rotate_reel_start_index))

    def _reel_step_visible(step_i):
        return (
            step_i % reel_render_stride == 0
            or step_i == n_steps - 1
        )

    if rotate_continuous and not rotate_reel_only:
        rot_total = _ch6_estimate_dist_reel_frame_count(
            pack,
            fast=fast,
            keep_classroom_on_stage=keep_classroom_on_stage,
            density_at_end=density_at_end,
            density_end_compact=density_end_compact,
            density_histograms=density_histograms,
            rotate_continuous=True,
            reel_render_stride=reel_render_stride,
        )
    elif rotate_reel_only:
        rot_total = sum(
            1 for step_i in range(n_steps)
            if step_i >= rotate_reel_start_index and _reel_step_visible(step_i)
        )
        rot_total = max(rot_total, 1)

    def _take_azim(*, rotate_now=False):
        nonlocal rot_i
        if rotate_now and rot_total > 0:
            az = base_azim + 360.0 * rot_i / max(rot_total - 1, 1)
            rot_i += 1
            return az
        return base_azim

    def _step_rotates(step_i):
        if rotate_reel_only:
            return step_i >= rotate_reel_start_index
        return rotate_continuous and not rotate_reel_only

    def _cloud_frame(*, rotating=False, **extra):
        if "view_azim" not in extra:
            extra["view_azim"] = _take_azim(
                rotate_now=rotating or (rotate_continuous and not rotate_reel_only),
            )
        return _frame_duo(
            xlim=xlim, ylim=ylim, pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
            **cloud_kw, **duo_kw, **extra,
        )

    def _append_cloud(*, rotating=False, **extra):
        nonlocal last_frame
        last_frame = _finish(_cloud_frame(rotating=rotating, **extra), clip_id)
        return last_frame

    def _hold_cloud(n, *, rotating=False, **extra):
        n = int(max(0, n))
        if n <= 0:
            return
        for _ in range(n):
            if extra and (
                (rotate_continuous and not rotate_reel_only)
                or rotating
            ):
                yield _append_cloud(rotating=rotating, **extra)
            elif last_frame is not None:
                yield last_frame

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
    open_kw = dict(
        base_study=open_s, base_exam=open_e, base_y=open_y,
        w_live=w_open, knob_w=w_open, show_ellipses=True,
        markers=np.asarray([w_open]), highlight_w=w_open,
    )
    yield _append_cloud(**open_kw)
    yield from _hold_cloud(n_hold_open, **open_kw)

    for u in np.linspace(0.0, 1.0, _draft_short(12, 5)):
        pop_kw = dict(
            open_kw,
            pop_alpha=0.04 + 0.14 * float(u),
            show_ellipses=(u < 0.35),
        )
        yield _append_cloud(**pop_kw)
    yield from _hold_cloud(n_hold_pop, **open_kw)

    match_kw = open_kw
    if match_s is not None:
        for u in np.linspace(0.0, 1.0, _draft_short(10, 4)):
            uu = float(u)
            match_kw = dict(
                base_study=(1.0 - uu) * open_s + uu * match_s,
                base_exam=(1.0 - uu) * open_e + uu * pack["match_e"],
                base_y=open_y,
                w_live=(1.0 - uu) * w_open + uu * w_seed,
                knob_w=(1.0 - uu) * w_open + uu * w_seed,
                markers=np.asarray([(1.0 - uu) * w_open + uu * w_seed]),
                highlight_w=(1.0 - uu) * w_open + uu * w_seed,
            )
            yield _append_cloud(**match_kw)
        yield from _hold_cloud(n_hold_pop, **match_kw)

    last_step = pack["reel_steps"][-1]
    for step_i, step in enumerate(pack["reel_steps"]):
        if not _reel_step_visible(step_i):
            continue
        rotating = _step_rotates(step_i)
        cs, ce, cy = step["study"], step["exam"], step["y"]
        w = step["w"]
        markers = step["markers"]
        ghosts = step["ghosts"]
        step_kw = dict(
            base_study=cs, base_exam=ce, base_y=cy,
            w_live=w, ghost_ws=ghosts, knob_w=w,
            markers=markers, highlight_w=w,
            **_density_kw(markers),
        )
        yield _append_cloud(rotating=rotating, **step_kw)
        yield from _hold_cloud(n_flash, rotating=rotating, **step_kw)
        if not keep_classroom_on_stage:
            clear_kw = dict(
                base_study=cs[:0], base_exam=ce[:0], base_y=cy[:0],
                show_base=False,
                ghost_ws=ghosts + [w], knob_w=w,
                markers=markers, highlight_w=w,
                **_density_kw(markers),
            )
            yield _append_cloud(rotating=rotating, **clear_kw)
            yield from _hold_cloud(n_seq_hold, rotating=rotating, **clear_kw)
        else:
            yield from _hold_cloud(n_seq_hold, rotating=rotating, **step_kw)

    # --- overlay sampling distribution on the final cloud ---
    cs, ce, cy = last_step["study"], last_step["exam"], last_step["y"]
    markers = last_step["markers"]
    ghosts = last_step["ghosts"]
    w = last_step["w"]

    def _final_frame(
        *,
        ell_alpha=0.0,
        show_x=False,
        azim=None,
        density_reveal_u=0.0,
        density_projection_u=0.0,
    ):
        use_azim = base_azim if azim is None else azim
        if rotate_continuous and not rotate_reel_only and azim is None:
            use_azim = _take_azim(rotate_now=True)
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
            view_azim=use_azim,
            **dens_kw,
        )

    def _append_final(**kwargs):
        nonlocal last_frame
        last_frame = _finish(_final_frame(**kwargs), clip_id)
        return last_frame

    def _hold_final(n, **kwargs):
        n = int(max(0, n))
        for _ in range(n):
            if rotate_continuous and not rotate_reel_only:
                yield _append_final(**kwargs)
            elif last_frame is not None:
                yield last_frame

    yield _append_final()
    yield from _hold_final(CH6_N_HOLD)

    if density_at_end:
        n_proj = _draft_short(16 if density_end_compact else 32, 8)
        for u in np.linspace(0.0, 1.0, n_proj):
            yield _append_final(density_projection_u=float(u), density_reveal_u=0.0)
        hold_d = CH6_N_HOLD if density_end_compact else CH6_N_HOLD * 2
        yield _append_final(density_projection_u=1.0, density_reveal_u=0.0)
        yield from _hold_final(hold_d, density_projection_u=1.0, density_reveal_u=0.0)
        n_rise = _draft_short(36 if density_end_compact else 72, 18)
        for u in np.linspace(0.0, 1.0, n_rise):
            uu = float(u) ** 1.35
            yield _append_final(density_projection_u=1.0, density_reveal_u=uu)
        yield _append_final(density_projection_u=1.0, density_reveal_u=1.0)
        yield from _hold_final(
            hold_d, density_projection_u=1.0, density_reveal_u=1.0,
        )
        if not rotate_continuous:
            n_spin = _draft_short(36 if density_end_compact else 72, 14)
            for t in range(n_spin + 1):
                az = base_azim + 360.0 * t / n_spin
                yield _append_final(
                    density_projection_u=1.0,
                    density_reveal_u=1.0,
                    azim=az,
                )
        yield from _hold_final(
            CH6_N_HOLD * 2 if not (rotate_continuous and not rotate_reel_only) else 1,
            density_projection_u=1.0,
            density_reveal_u=1.0,
        )
        return

    if density_histograms:
        yield _append_final(density_reveal_u=1.0)
        yield from _hold_final(CH6_N_HOLD * 2, density_reveal_u=1.0)
        if density_show_mean_x:
            for u in np.linspace(0.0, 1.0, _draft_short(8, 4)):
                yield _append_final(density_reveal_u=1.0, show_x=(u > 0.2))
            yield _append_final(density_reveal_u=1.0, show_x=True)
            yield from _hold_final(CH6_N_HOLD * 2, density_reveal_u=1.0, show_x=True)
        n_spin = _draft_short(72, 28)
        for t in range(n_spin + 1):
            az = base_azim + 360.0 * t / n_spin
            yield _append_final(
                density_reveal_u=1.0,
                show_x=density_show_mean_x,
                azim=az,
            )
        yield from _hold_final(CH6_N_HOLD * 2, density_reveal_u=1.0, show_x=density_show_mean_x)
        return

    for u in np.linspace(0.0, 1.0, _draft_short(14, 6)):
        yield _append_final(ell_alpha=float(u))
    for u in np.linspace(0.0, 1.0, _draft_short(8, 4)):
        yield _append_final(ell_alpha=1.0, show_x=(u > 0.2))
    yield _append_final(ell_alpha=1.0, show_x=True)
    yield from _hold_final(CH6_N_HOLD * 2, ell_alpha=1.0, show_x=True)

    n_spin = _draft_short(72, 28)
    for t in range(n_spin + 1):
        az = base_azim + 360.0 * t / n_spin
        yield _append_final(ell_alpha=1.0, show_x=True, azim=az)
    yield from _hold_final(CH6_N_HOLD * 2, ell_alpha=1.0, show_x=True)


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


def build_ch6_97_population_reel_cloud_d1_2d_only(clip_id):
    """Like ch6_68 but full-screen 2D only — no knobs, Gaussians, or 3D cloud."""
    return _build_population_classroom_reel_param_cloud(
        clip_id,
        n_class=CH6_N_CLASS_BASE,
        seed=17,
        seed_from_key="D1",
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True,
        wide_2d_only=True,
        class_mean_blend=0.0,
    )


def _ch6_reel_panel_kwargs(pack, step_idx, *, show_pop=False):
    idx = int(np.clip(int(step_idx), 0, len(pack["reel_steps"]) - 1))
    step = pack["reel_steps"][idx]
    pop = (
        (pack["pop_s"], pack["pop_e"], pack["pop_y"])
        if show_pop else (None, None, None)
    )
    return dict(
        base_study=step["study"],
        base_exam=step["exam"],
        base_y=step["y"],
        left_w=step["w"],
        left_ghosts=list(step["ghosts"]),
        left_pop_study=pop[0],
        left_pop_exam=pop[1],
        left_pop_y=pop[2],
        left_pop_alpha=0.0,
    )


def _ch6_reel_right_panel_kwargs(pack, step_idx, *, show_pop=False):
    idx = int(np.clip(int(step_idx), 0, len(pack["reel_steps"]) - 1))
    step = pack["reel_steps"][idx]
    pop = (
        (pack["pop_s"], pack["pop_e"], pack["pop_y"])
        if show_pop else (None, None, None)
    )
    return dict(
        right_base_study=step["study"],
        right_base_exam=step["exam"],
        right_base_y=step["y"],
        right_w=step["w"],
        right_ghosts=list(step["ghosts"]),
        right_pop_study=pop[0],
        right_pop_exam=pop[1],
        right_pop_y=pop[2],
        right_pop_alpha=0.0,
    )


def _ch6_dual_panel_98_config():
    """Shared packs + end indices for ch6_98 / ch6_99 handoff continuity."""
    n_right_dual = _draft_short(44, 14) * 4
    n_morph = int(CH5_BEST_LINE_N_LAYOUT)
    tick_period = _ch6_reel_tick_period()
    n_right_in = _draft_short(10, 4)
    n_left_extra = (n_morph // tick_period) + (n_right_dual // 2) + n_right_dual + 16

    left_pack = _population_param_cloud_landings(
        n_class=CH6_N_CLASS_BASE,
        seed=17,
        seed_from_key="D1",
        class_mean_blend=0.0,
        n_reel=int(CH6_N_POP_REEL) + int(n_left_extra),
    )
    right_pack = _population_param_cloud_landings(
        n_class=CH6_N_CLASS_BASE,
        seed=97,
        seed_from_key="D1",
        n_reel=int(n_right_dual),
        pop_mu_fail=CH6_WILD_MU_FAIL,
        pop_mu_pass=CH6_WILD_MU_PASS,
        pop_cov_fail=CH6_WILD_COV_FAIL,
        pop_cov_pass=CH6_WILD_COV_PASS,
        pop_seed_offset=31,
    )

    i_mid = len(left_pack["reel_steps"]) // 2
    li = i_mid
    tick_acc = 0
    for _ in range(max(1, n_morph)):
        tick_acc += 1
        if tick_acc >= tick_period:
            li += 1
            tick_acc = 0
    for _ in range(max(1, n_right_in)):
        tick_acc += 1
        if tick_acc >= tick_period:
            li += 1
            tick_acc = 0
    li_end = li + n_right_dual - 1
    ri_end = n_right_dual - 1
    return left_pack, right_pack, li_end, ri_end, n_right_dual, n_morph, n_right_in


def _build_population_reel_d1_dual_panel_handoff(clip_id):
    """Mid ch6_97 → resize left 2D; wild right reel; synchronized dual wiggles."""
    left_pack, right_pack, _, _, n_right_dual, n_morph, n_right_in = _ch6_dual_panel_98_config()
    tick_period = _ch6_reel_tick_period()
    xlim, ylim = left_pack["xlim"], left_pack["ylim"]
    i_mid = len(left_pack["reel_steps"]) // 2
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    n_tick_hold = CH6_N_FLASH + CH6_N_SEQ_HOLD

    def _dual_frame(layout_u, right_u, li, ri=None, *, show_right=False):
        kw = _ch6_reel_panel_kwargs(left_pack, li, show_pop=False)
        if show_right and ri is not None:
            kw.update(_ch6_reel_right_panel_kwargs(right_pack, ri, show_pop=False))
        return _frame_ch6_dual_reel_2d(
            layout_u, right_u,
            xlim=xlim, ylim=ylim,
            left_base_study=kw["base_study"],
            left_base_exam=kw["base_exam"],
            left_base_y=kw["base_y"],
            left_pop_study=None,
            left_pop_exam=None,
            left_pop_y=None,
            left_pop_alpha=0.0,
            left_w=kw["left_w"],
            left_ghosts=kw["left_ghosts"],
            right_base_study=kw.get("right_base_study"),
            right_base_exam=kw.get("right_base_exam"),
            right_base_y=kw.get("right_base_y"),
            right_pop_study=None,
            right_pop_exam=None,
            right_pop_y=None,
            right_pop_alpha=0.0,
            right_w=kw.get("right_w"),
            right_ghosts=kw.get("right_ghosts"),
            show_right=show_right,
        )

    def _maybe_advance(li, frames_since_tick):
        frames_since_tick += 1
        if frames_since_tick >= tick_period:
            return li + 1, 0
        return li, frames_since_tick

    frames = []
    li = i_mid
    mid_kw = _ch6_reel_panel_kwargs(left_pack, li, show_pop=False)
    img = _frame_ch6_fullscreen_2d(
        xlim=xlim, ylim=ylim,
        base_study=mid_kw["base_study"],
        base_exam=mid_kw["base_exam"],
        base_y=mid_kw["base_y"],
        pop_study=None,
        pop_exam=None,
        pop_y=None,
        w_live=mid_kw["left_w"],
        ghost_ws=mid_kw["left_ghosts"],
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], n_hold))

    tick_acc = 0
    for j in range(max(1, n_morph)):
        u = smooth(float(j) / max(n_morph - 1, 1))
        frames.append(_finish(_dual_frame(u, 0.0, li), clip_id))
        li, tick_acc = _maybe_advance(li, tick_acc)

    ri = 0
    for j in range(max(1, n_right_in)):
        ru = smooth(float(j + 1) / max(n_right_in, 1))
        frames.append(_finish(_dual_frame(1.0, ru, li, ri, show_right=True), clip_id))
        li, tick_acc = _maybe_advance(li, tick_acc)

    li_start_dual = li
    tick_acc = 0
    for k in range(n_right_dual):
        li = li_start_dual + k
        frames.append(_finish(_dual_frame(1.0, 1.0, li, k, show_right=True), clip_id))
        frames.extend(_hold(frames[-1], n_tick_hold))

    frames.extend(_hold(frames[-1], n_hold))
    return frames


def build_ch6_98_population_reel_d1_wide_to_duo(clip_id):
    return _build_population_reel_d1_dual_panel_handoff(clip_id)


def _build_population_reel_split_to_duo_handoff(clip_id):
    """Continue from ch6_98 end: split panels → ch6_68 duo (D1, knobs, empty 3D)."""
    left_pack, right_pack, li_end, ri_end, _, _, _ = _ch6_dual_panel_98_config()
    end_l = _ch6_reel_panel_kwargs(left_pack, li_end, show_pop=False)
    end_r = _ch6_reel_right_panel_kwargs(right_pack, ri_end, show_pop=False)

    d1_s, d1_e, d1_y = ch5_unpack_dataset("D1")
    d1_s = np.asarray(d1_s, dtype=np.float64)
    d1_e = np.asarray(d1_e, dtype=np.float64)
    d1_y = np.asarray(d1_y, dtype=np.float64)
    w_d1, _ = ch6_fit_population_classroom(d1_s, d1_e, d1_y)

    end_cs = np.asarray(end_l["base_study"], dtype=np.float64)
    end_ce = np.asarray(end_l["base_exam"], dtype=np.float64)
    end_cy = np.asarray(end_l["base_y"], dtype=np.float64)
    end_w = np.asarray(end_l["left_w"], dtype=np.float64)
    end_ghosts = list(end_l["left_ghosts"])

    xlim, ylim = left_pack["xlim"], left_pack["ylim"]
    frames = []
    n_hold = CH6_N_HOLD * 2
    n_layout = int(CH5_BEST_LINE_N_LAYOUT)
    n_fade = _draft_short(12, 4)
    smooth = _g("ch3_knob_smoothstep")

    end_kw = _ch6_reel_panel_kwargs(left_pack, li_end, show_pop=False)
    end_rkw = _ch6_reel_right_panel_kwargs(right_pack, ri_end, show_pop=False)
    img = _frame_ch6_dual_reel_2d(
        1.0, 1.0,
        xlim=xlim, ylim=ylim,
        left_base_study=end_kw["base_study"],
        left_base_exam=end_kw["base_exam"],
        left_base_y=end_kw["base_y"],
        left_w=end_kw["left_w"],
        left_ghosts=end_kw["left_ghosts"],
        right_base_study=end_rkw["right_base_study"],
        right_base_exam=end_rkw["right_base_exam"],
        right_base_y=end_rkw["right_base_y"],
        right_w=end_rkw["right_w"],
        right_ghosts=end_rkw["right_ghosts"],
        show_right=True,
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], n_hold))

    for i in range(max(1, n_layout)):
        u = smooth(float(i) / max(n_layout - 1, 1))
        fade = 1.0 - u
        bs = (1.0 - u) * end_cs + u * d1_s
        be = (1.0 - u) * end_ce + u * d1_e
        ww = (1.0 - u) * end_w + u * w_d1
        ghosts = end_ghosts if fade > 0.02 else []
        img = _frame_ch6_split_to_duo_empty3d(
            u,
            right_fade_u=fade,
            xlim=xlim, ylim=ylim,
            base_study=bs, base_exam=be, base_y=d1_y,
            w_live=ww, ghost_ws=ghosts, ghost_fade_u=fade,
            knob_w=ww, show_3d_u=0.0,
            right_base_study=end_rkw["right_base_study"] if fade > 0.02 else None,
            right_base_exam=end_rkw["right_base_exam"] if fade > 0.02 else None,
            right_base_y=end_rkw["right_base_y"] if fade > 0.02 else None,
            right_w=end_rkw["right_w"] if fade > 0.02 else None,
            right_ghosts=end_rkw["right_ghosts"] if fade > 0.02 else None,
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], n_hold))

    duo_no3d = frames[-1]
    duo_with3d = _finish(
        _frame_ch6_split_to_duo_empty3d(
            1.0,
            right_fade_u=0.0,
            xlim=xlim, ylim=ylim,
            base_study=d1_s, base_exam=d1_e, base_y=d1_y,
            w_live=w_d1, ghost_ws=[], ghost_fade_u=0.0,
            knob_w=w_d1, show_3d_u=1.0,
        ),
        clip_id,
    )
    for i in range(max(1, n_fade)):
        u = smooth(float(i + 1) / max(n_fade, 1))
        frames.append(ch5_crossfade_images(duo_no3d, duo_with3d, u))
    frames.extend(_hold(frames[-1], n_hold))
    return frames


def build_ch6_99_population_reel_d1_split_to_duo(clip_id):
    return _build_population_reel_split_to_duo_handoff(clip_id)


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
    cloud_kw = dict(
        marker_z_mode="bias", marker_axis_lim=axis_lim, zlabel="b",
        marker_density_reveal_u=1.0,
    )

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
    return _population_reel_param_cloud_dist_frames(
        clip_id, n_class=CH6_N_CLASS_BASE, seed=17, seed_from_key="D1",
        n_reel=CH6_DENSITY_END_D1_N_REEL,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True, density_at_end=True,
        fast=True, density_end_compact=True,
        reel_render_stride=10,
    )


def build_ch6_101_population_reel_cloud_dist_d1_density_end_spin(clip_id):
    """Like ch6_89 but 360° azimuth sweep from the 2nd reel dataset through the last."""
    return _population_reel_param_cloud_dist_frames(
        clip_id, n_class=CH6_N_CLASS_BASE, seed=17, seed_from_key="D1",
        n_reel=CH6_DENSITY_END_D1_N_REEL,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True, density_at_end=True,
        fast=True, density_end_compact=True,
        rotate_reel_only=True,
        rotate_reel_start_index=1,
        reel_render_stride=10,
    )


def _build_cloud_rotate_warp_from_density_end(
    clip_id, *, show_box=False, use_density_colors=True,
):
    """ch6_89 cloud (no histograms) → 360° spin → 720° warp reel through shapes."""
    pack = _ch6_density_end_d1_pack()
    W_orig = np.asarray(pack["reel_steps"][-1]["markers"], dtype=np.float64)
    axis_lim = CH6_POPULATION_PARAM_AXIS_LIM
    targets = _ch6_warp_target_clouds(W_orig, axis_lim)
    path = (
        targets["original"],
        targets["isotropic"],
        targets["bimodal"],
        targets["elongated"],
        targets["original"],
    )

    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    n_spin1 = _draft_short(72, 28)
    n_leg = _draft_short(32, 12)
    warp_az_total = 720.0
    box_bounds = (
        (float(np.min(W_orig[:, 0])), float(np.max(W_orig[:, 0]))),
        (float(np.min(W_orig[:, 1])), float(np.max(W_orig[:, 1]))),
        (float(np.min(W_orig[:, 2])), float(np.max(W_orig[:, 2]))),
    )
    box_kw = (
        dict(show_variance_box=True, variance_box_bounds=box_bounds)
        if show_box else {}
    )
    density_kw = CH6_BELIEF_DENSITY_KW if use_density_colors else {}

    def emit(W, azim):
        return _finish(
            _frame_duo(
                xlim=pack["xlim"],
                ylim=pack["ylim"],
                pop_study=pack["pop_s"],
                pop_exam=pack["pop_e"],
                pop_y=pack["pop_y"],
                pop_alpha=0.08,
                base_study=pack["reel_steps"][-1]["study"],
                base_exam=pack["reel_steps"][-1]["exam"],
                base_y=pack["reel_steps"][-1]["y"],
                w_live=pack["reel_steps"][-1]["w"],
                knob_w=pack["reel_steps"][-1]["w"],
                highlight_w=pack["reel_steps"][-1]["w"],
                ghost_ws=pack["reel_steps"][-1]["ghosts"],
                markers=W,
                view_azim=azim,
                marker_z_mode="bias",
                marker_axis_lim=axis_lim,
                zlabel="b",
                show_legend=False,
                title_left=None,
                title_right=None,
                **density_kw,
                **box_kw,
            ),
            clip_id,
        )

    frames: list = []
    frames.extend(_hold(emit(W_orig, base_azim), n_hold))

    for t in range(1, n_spin1 + 1):
        az = base_azim + 360.0 * t / n_spin1
        frames.append(emit(W_orig, az))

    n_legs = len(path) - 1
    total_morph = n_legs * n_leg
    for leg in range(n_legs):
        a = np.asarray(path[leg], dtype=np.float64)
        b = np.asarray(path[leg + 1], dtype=np.float64)
        for j in range(n_leg):
            u = smooth(float(j) / max(n_leg - 1, 1))
            W = (1.0 - u) * a + u * b
            frame_i = leg * n_leg + j
            az = base_azim + 360.0 + warp_az_total * (frame_i + 1) / max(total_morph, 1)
            frames.append(emit(W, az))

    frames.extend(_hold(frames[-1], n_hold))
    return frames


def build_ch6_100_population_cloud_d1_rotate_warp(clip_id):
    return _build_cloud_rotate_warp_from_density_end(clip_id)


def build_ch6_109_population_cloud_d1_rotate_warp_box(clip_id):
    """Duplicate of ch6_100 with a fixed 3D bounds box around the cloud."""
    return _build_cloud_rotate_warp_from_density_end(clip_id, show_box=True)


def build_ch6_110_population_cloud_d1_rotate_warp_box_plain(clip_id):
    """Duplicate of ch6_109 with fixed box and plain (non-density) cloud colors."""
    return _build_cloud_rotate_warp_from_density_end(
        clip_id, show_box=True, use_density_colors=False,
    )


def build_ch6_111_population_cloud_d1_rotate_warp_box_plain_planar(clip_id):
    """Planar warp: each target is constrained to the cloud's best-fit plane."""
    pack = _ch6_density_end_d1_pack()
    last = pack["reel_steps"][-1]
    W_orig = np.asarray(last["markers"], dtype=np.float64)
    axis_lim = CH6_POPULATION_PARAM_AXIS_LIM
    targets = _ch6_planar_warp_target_clouds(W_orig, seed=111)
    path = (
        targets["original"],
        targets["isotropic"],
        targets["bimodal"],
        targets["elongated"],
        targets["original"],
    )

    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    n_spin1 = _draft_short(72, 28)
    n_warp = _draft_short(18, 8)
    n_freeze = 3 * n_warp  # 270° hold after each 90° warp

    def emit(W, azim):
        return _finish(
            _frame_duo(
                xlim=pack["xlim"],
                ylim=pack["ylim"],
                pop_study=pack["pop_s"],
                pop_exam=pack["pop_e"],
                pop_y=pack["pop_y"],
                pop_alpha=0.08,
                base_study=last["study"],
                base_exam=last["exam"],
                base_y=last["y"],
                w_live=last["w"],
                knob_w=last["w"],
                highlight_w=last["w"],
                ghost_ws=last["ghosts"],
                markers=W,
                view_azim=azim,
                marker_z_mode="bias",
                marker_axis_lim=axis_lim,
                zlabel="b",
                show_legend=False,
                title_left=None,
                title_right=None,
            ),
            clip_id,
        )

    frames: list = []
    frames.extend(_hold(emit(W_orig, base_azim), n_hold))
    for t in range(1, n_spin1 + 1):
        az = base_azim + 360.0 * t / n_spin1
        frames.append(emit(W_orig, az))

    az_cursor = base_azim + 360.0
    n_legs = len(path) - 1
    for leg in range(n_legs):
        a = np.asarray(path[leg], dtype=np.float64)
        b = np.asarray(path[leg + 1], dtype=np.float64)
        for j in range(n_warp):
            u = smooth(float(j) / max(n_warp - 1, 1))
            W = (1.0 - u) * a + u * b
            az = az_cursor + 90.0 * float(j + 1) / max(n_warp, 1)
            frames.append(emit(W, az))
        az_cursor += 90.0
        for j in range(n_freeze):
            az = az_cursor + 270.0 * float(j + 1) / max(n_freeze, 1)
            frames.append(emit(b, az))
        az_cursor += 270.0

    frames.extend(_hold(frames[-1], n_hold))
    return frames


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
CH6_TRUE_LABEL_COLOR = CH6_VARIANCE_RED
CH6_ESTIMATOR_LABEL_COLOR = _CH6_GRAD_VECTOR_COLOR


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
        ax.quiver(
            study[mask], exam[mask], U[mask], V[mask],
            angles="xy", scale_units="xy", scale=1.0,
        width=shaft_w,
        headwidth=head_w,
        headlength=head_l,
        headaxislength=head_a,
        color=color,
        alpha=float(alpha),
        pivot="tail",
        zorder=9,
        minshaft=1.5,
        minlength=0.4,
    )


def _draw_point_grad_quivers_away_from_line_2d(
    ax,
    study,
    exam,
    contrib,
    w,
    *,
    color=None,
    span_frac=0.22,
    alpha=0.98,
    compact=False,
    highlight_idx=None,
    highlight_color="#e74c3c",
    highlight_max_ratio=3.8,
):
    """Per-point ∇NLL projected onto the direction from each point away from the line.

    Magnitude is ``|g·d̂|``; direction is always away from the boundary on the
    point's side (so separable pass/fail both push outward on the line).
    """
    color = _CH6_GRAD_VECTOR_COLOR if color is None else color
    study = np.asarray(study, dtype=np.float64).reshape(-1)
    exam = np.asarray(exam, dtype=np.float64).reshape(-1)
    n = len(study)
    G = np.asarray(contrib, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(3)
    if G.ndim != 2 or G.shape[0] == 0:
        return
    a, b, c = float(w[0]), float(w[1]), float(w[2])
    den = a * a + b * b
    if den < 1e-12:
        return
    z = a * study + b * exam + c
    t = -z / den
    foot_s = study + t * a
    foot_e = exam + t * b
    away_s = study - foot_s
    away_e = exam - foot_e
    dist = np.hypot(away_s, away_e)
    mask = dist > 1e-4
    if not np.any(mask):
        return

    G2 = G[:, :2]
    proj = np.zeros(len(study), dtype=np.float64)
    d_hat = np.zeros((len(study), 2), dtype=np.float64)
    for i in np.where(mask)[0]:
        d_hat[i] = away_s[i] / dist[i], away_e[i] / dist[i]
        proj[i] = abs(float(G2[i] @ d_hat[i]))

    peak_all = float(np.max(proj[mask])) + 1e-12
    hi = int(highlight_idx) if highlight_idx is not None else -1
    roster_ix = [i for i in np.where(mask)[0] if int(i) != hi]
    if hi >= 0 and hi < n and roster_ix:
        peak_ref = float(np.max(proj[roster_ix])) + 1e-12
        hi_cap = float(highlight_max_ratio)
    else:
        peak_ref = peak_all
        hi_cap = 1.0

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    target = float(span_frac) * min(float(xlim[1] - xlim[0]), float(ylim[1] - ylim[0]))
    if compact:
        target *= 0.72
        shaft_w = 0.0065
        head_w, head_l, head_a = 3.4, 4.0, 3.0
    elif n <= 12:
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
    for i in np.where(mask)[0]:
        mag = float(proj[i])
        if mag < 1e-9:
            continue
        ratio = mag / peak_ref
        if int(i) == hi:
            length = min(ratio, hi_cap) * target
        else:
            length = min(ratio, 1.0) * target
        if length < 0.05 * target:
            continue
        U[i] = length * float(d_hat[i, 0])
        V[i] = length * float(d_hat[i, 1])

    draw_mask = (np.abs(U) + np.abs(V)) > 0
    if not np.any(draw_mask):
        return
    for i in np.where(draw_mask)[0]:
        col = highlight_color if int(i) == hi else color
        sw = shaft_w * (1.30 if int(i) == hi else 1.0)
        ax.quiver(
            study[i], exam[i], U[i], V[i],
            angles="xy", scale_units="xy", scale=1.0,
            width=sw,
            headwidth=head_w,
            headlength=head_l,
            headaxislength=head_a,
            color=col,
            alpha=float(alpha),
            pivot="tail",
            zorder=9,
            minshaft=1.5,
            minlength=0.4,
        )


def _draw_point_arrows_toward_line_2d(
    ax,
    study,
    exam,
    w,
    *,
    color=None,
    span_frac=0.20,
    lw=2.4,
    alpha=0.98,
    highlight_idx=None,
    highlight_color="#e74c3c",
):
    """Arrows from the decision boundary to each student (exact perpendicular span)."""
    color = _CH6_GRAD_VECTOR_COLOR if color is None else color
    study = np.asarray(study, dtype=np.float64).reshape(-1)
    exam = np.asarray(exam, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(3)
    a, b, c = float(w[0]), float(w[1]), float(w[2])
    den = a * a + b * b
    if den < 1e-12 or len(study) == 0:
        return
    z = a * study + b * exam + c
    t = -z / den
    foot_s = study + t * a
    foot_e = exam + t * b
    U = study - foot_s
    V = exam - foot_e
    dist = np.hypot(U, V)
    mask = dist > 1e-4
    if not np.any(mask):
        return
    n = len(study)
    if n <= 12:
        shaft_w = 0.0075
        head_w, head_l, head_a = 4.0, 4.5, 3.5
    else:
        shaft_w = 0.006
        head_w, head_l, head_a = 3.5, 4.0, 3.2

    hi = int(highlight_idx) if highlight_idx is not None else -1
    for i in np.where(mask)[0]:
        col = highlight_color if i == hi else color
        ax.quiver(
            foot_s[i], foot_e[i], U[i], V[i],
            angles="xy", scale_units="xy", scale=1.0,
            width=shaft_w,
            headwidth=head_w,
            headlength=head_l,
            headaxislength=head_a,
            color=col,
            alpha=float(alpha),
            pivot="tail",
            zorder=8,
            minshaft=1.0,
            minlength=0.2,
        )


def _ch6_d1_balanced_six():
    """Six D1 students: three fail, three pass (one fail near study=2, exam=4.5)."""
    s, e, y = ch5_unpack_dataset("D1")
    s = np.asarray(s, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    fail_i = [i for i in range(len(y)) if int(y[i]) == 0]
    pass_i = [i for i in range(len(y)) if int(y[i]) == 1]
    anchor = (2.0, 4.5)
    anchor_fail = min(
        fail_i,
        key=lambda i: float((s[i] - anchor[0]) ** 2 + (e[i] - anchor[1]) ** 2),
    )
    other_fails = [i for i in fail_i if i != anchor_fail][:2]
    pick = [anchor_fail] + other_fails + pass_i[:3]
    return s[pick], e[pick], y[pick]


def _ch6_roster_indices_for_points(study, exam, y, pick_s, pick_e, pick_y):
    """Pick roster rows that best match reference (study, exam, label) points."""
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    pick_s = np.asarray(pick_s, dtype=np.float64)
    pick_e = np.asarray(pick_e, dtype=np.float64)
    pick_y = np.asarray(pick_y, dtype=np.int64)
    used: set[int] = set()
    out: list[int] = []

    def _best_for(ps, pe, py, *, require_label):
        best_i = None
        best_d = np.inf
        for i in range(len(study)):
            if i in used:
                continue
            if require_label and int(y[i]) != int(py):
                continue
            d = float((study[i] - ps) ** 2 + (exam[i] - pe) ** 2)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    for ps, pe, py in zip(pick_s, pick_e, pick_y):
        hit = _best_for(ps, pe, py, require_label=True)
        if hit is None:
            hit = _best_for(ps, pe, py, require_label=False)
        if hit is not None:
            used.add(hit)
            out.append(hit)
    return out


def _ch6_loo_n6_peel_setup(cs0, ce0, cy0, *, n_target=6):
    """Return ``(remove_order, keep_idx)`` for peeling a roster down to ``n_target``."""
    ref_s, ref_e, ref_y = _ch6_d1_balanced_six()
    keep_idx = set(_ch6_roster_indices_for_points(cs0, ce0, cy0, ref_s, ref_e, ref_y))
    if len(keep_idx) < n_target:
        for i in range(len(cs0)):
            if i not in keep_idx:
                keep_idx.add(i)
                if len(keep_idx) >= n_target:
                    break
    remove_order = [i for i in range(len(cs0) - 1, -1, -1) if i not in keep_idx]
    return remove_order, keep_idx


def _build_cloud_loo_down_n6_spring(clip_id, *, grad_arrows=False):
    """Cloud hold → fade ghosts → peel roster to six D1 → arrows → spring → outlier."""
    pack = _ch6_density_end_d1_pack()
    W_orig = np.asarray(pack["reel_steps"][-1]["markers"], dtype=np.float64)
    axis_lim = CH6_POPULATION_PARAM_AXIS_LIM
    xlim, ylim = pack["xlim"], pack["ylim"]
    last = pack["reel_steps"][-1]
    cs0 = np.asarray(last["study"], dtype=np.float64)
    ce0 = np.asarray(last["exam"], dtype=np.float64)
    cy0 = np.asarray(last["y"], dtype=np.int64)
    pop_s, pop_e, pop_y = pack["pop_s"], pack["pop_e"], pack["pop_y"]

    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))
    n_hold = CH6_N_HOLD * 2
    duo_pop = dict(
        pop_alpha=0.08, show_legend=False, marker_z_mode="bias",
        marker_axis_lim=axis_lim, zlabel="b", ghost_fade_u=0.0,
        title_left=None, title_right=None,
    )
    cloud_kw = dict(
        marker_z_mode="bias", marker_axis_lim=axis_lim, zlabel="b",
        title_left=None, title_right=None,
        **CH6_BELIEF_DENSITY_KW,
    )

    def emit_cloud(W, azim, *, ghost_fade_u=1.0, ghosts=None):
        return _finish(
            _ch6_cloud_frame_from_pack(
                pack, markers=W, view_azim=azim, ghost_fade_u=ghost_fade_u,
                **CH6_BELIEF_DENSITY_KW,
            ),
            clip_id,
        )

    peel_clouds = _ch6_d1_peel_marker_clouds_cached()

    def emit_roster(cs, ce, cy, w, azim, *, markers, ghost_ws=None, ghost_fade_u=0.0):
        n_class = int(len(cs))
        marker_s = ch6_population_n_sweep_marker_size(n_class)
        highlight_marker_s = marker_s * (70.0 / 28.0)
        return _finish(
            _frame_duo(
                xlim=xlim, ylim=ylim,
                pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
                pop_alpha=0.08,
                base_study=cs, base_exam=ce, base_y=cy,
                w_live=w, knob_w=w, highlight_w=w,
                markers=markers, view_azim=azim,
                marker_s=marker_s, highlight_marker_s=highlight_marker_s,
                ghost_ws=ghost_ws or [],
                ghost_fade_u=ghost_fade_u,
                show_legend=False,
                **cloud_kw,
            ),
            clip_id,
        )

    def emit_six(**kw):
        return _finish(_frame_duo(xlim=xlim, ylim=ylim, **duo_pop, **kw), clip_id)

    def arrow_kw(*, highlight=None):
        if grad_arrows:
            return dict(
                show_point_grads_away_from_line=True,
                point_grad_highlight=highlight,
            )
        return dict(
            show_point_grads_toward_line=True,
            line_to_point_arrow_highlight=highlight,
        )

    frames: list = []
    w_full = np.asarray(last["w"], dtype=np.float64)

    # --- Phase 1: hold opening frame (n=20 cloud + classroom) ---
    frames.extend(_hold(emit_cloud(W_orig, base_azim), n_hold))

    # --- Phase 2: fade ghost lines in 2D ---
    for u in np.linspace(1.0, 0.0, _draft_short(14, 6)):
        frames.append(emit_cloud(W_orig, base_azim, ghost_fade_u=float(u)))
    frames.extend(_hold(emit_cloud(W_orig, base_azim, ghost_fade_u=0.0), n_hold))

    n_target = 6
    # --- Phase 3: peel students one-by-one until six remain ---
    remove_order, keep_idx = _ch6_loo_n6_peel_setup(cs0, ce0, cy0, n_target=n_target)
    visible = np.ones(len(cs0), dtype=bool)
    w_live = w_full.copy()
    for rem_i in remove_order:
        if int(visible.sum()) <= n_target:
            break
        visible[rem_i] = False
        cs = cs0[visible]
        ce = ce0[visible]
        cy = cy0[visible]
        w_live, _ = ch6_fit_dataset(cs, ce, cy)
        n_vis = int(visible.sum())
        frames.append(emit_roster(
            cs, ce, cy, w_live, base_azim,
            markers=peel_clouds[n_vis],
        ))
    frames.extend(_hold(frames[-1], n_hold))

    # --- Phase 4: six students, no boundary line ---
    s6 = cs0[visible]
    e6 = ce0[visible]
    y6 = cy0[visible]
    w6, _ = ch6_fit_dataset(s6, e6, y6)
    frames.extend(_hold(emit_six(
        base_study=s6, base_exam=e6, base_y=y6,
        w_live=None, knob_w=w6,
        markers=np.asarray([w6], dtype=np.float64),
        view_azim=base_azim,
    ), n_hold))

    # --- Phase 5: line + per-student arrows ---
    frames.extend(_hold(emit_six(
        base_study=s6, base_exam=e6, base_y=y6,
        w_live=w6, knob_w=w6, highlight_w=w6,
        markers=np.asarray([w6], dtype=np.float64),
        view_azim=base_azim,
        **arrow_kw(),
    ), n_hold * 2))

    # --- Phase 6: nudge line; arrows stretch; spring back ---
    Xd6 = ch6_design(s6, e6)
    H6 = np.asarray(
        ch6_observed_information(w6, Xd6, y6, ridge=CH6_RIDGE)[:2, :2],
        dtype=np.float64,
    )
    w_opt = np.asarray(w6, dtype=np.float64).copy()
    nudge = np.array([1.85, -1.55], dtype=np.float64)
    lo1, hi1, lo2, hi2 = CH6_VIEW_BOUNDS_W12
    w_push = w_opt.copy()
    w_push[0] = float(np.clip(w_opt[0] + nudge[0], lo1, hi1))
    w_push[1] = float(np.clip(w_opt[1] + nudge[1], lo2, hi2))

    for u in np.linspace(0.0, 1.0, _draft_short(12, 5)):
        ww = (1.0 - float(u)) * w_opt + float(u) * w_push
        frames.append(emit_six(
            base_study=s6, base_exam=e6, base_y=y6,
            w_live=ww, knob_w=ww, highlight_w=ww,
            markers=np.asarray([w_opt, ww], dtype=np.float64),
            view_azim=base_azim,
            **arrow_kw(),
        ))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))

    for xy in _hessian_restore_path(
        w_push[:2], w_opt[:2], H6,
        n_frames=_draft_short(24, 10),
        damp=1.15, stiffness=5.0,
    )[1:]:
        ww = w_opt.copy()
        ww[0], ww[1] = float(xy[0]), float(xy[1])
        frames.append(emit_six(
            base_study=s6, base_exam=e6, base_y=y6,
            w_live=ww, knob_w=ww, highlight_w=ww,
            markers=np.asarray([w_opt, ww], dtype=np.float64),
            view_azim=base_azim,
            **arrow_kw(),
        ))
    frames.extend(_hold(frames[-1], n_hold))

    # --- Phase 7: outlier appears first; its arrow pulls; then the line moves ---
    s_out, e_out, y_out = 5.0, 2.0, 0
    s7 = np.append(s6, s_out)
    e7 = np.append(e6, e_out)
    y7 = np.append(y6, y_out)
    w7, _ = ch6_fit_dataset(s7, e7, y7)
    outlier_i = len(s6)

    frames.extend(_hold(emit_six(
        base_study=s7, base_exam=e7, base_y=y7,
        w_live=w_opt, knob_w=w_opt, highlight_w=w_opt,
        markers=np.asarray([w_opt], dtype=np.float64),
        view_azim=base_azim,
        **arrow_kw(highlight=outlier_i),
    ), n_hold * 2))

    for u in np.linspace(0.0, 1.0, _draft_short(18, 8)):
        uu = float(u)
        ww = (1.0 - uu) * w_opt + uu * w7
        frames.append(emit_six(
            base_study=s7, base_exam=e7, base_y=y7,
            w_live=ww, knob_w=ww, highlight_w=ww,
            markers=np.asarray([w_opt, w7], dtype=np.float64),
            view_azim=base_azim,
            **arrow_kw(highlight=outlier_i),
        ))

    frames.extend(_hold(emit_six(
        base_study=s7, base_exam=e7, base_y=y7,
        w_live=w7, knob_w=w7, highlight_w=w7,
        markers=np.asarray([w_opt, w7], dtype=np.float64),
        view_azim=base_azim,
        **arrow_kw(highlight=outlier_i),
    ), n_hold * 3))

    tag = "ch6_103" if grad_arrows else "ch6_102"
    print(f"  {tag}: done — {len(frames)} frames", flush=True)
    return frames


def build_ch6_102_cloud_loo_down_n6_spring(clip_id):
    return _build_cloud_loo_down_n6_spring(clip_id, grad_arrows=False)


def _build_ch6_103_cloud_loo_down_n6_spring_grads(clip_id):
    """Fullscreen peel + grad arrows; closing duo n-sweep with rotating 3D cloud."""
    pack = _ch6_density_end_d1_pack()
    W_orig = np.asarray(pack["reel_steps"][-1]["markers"], dtype=np.float64)
    axis_lim = CH6_POPULATION_PARAM_AXIS_LIM
    xlim, ylim = pack["xlim"], pack["ylim"]
    last = pack["reel_steps"][-1]
    cs0 = np.asarray(last["study"], dtype=np.float64)
    ce0 = np.asarray(last["exam"], dtype=np.float64)
    cy0 = np.asarray(last["y"], dtype=np.int64)
    pop_s, pop_e, pop_y = pack["pop_s"], pack["pop_e"], pack["pop_y"]
    ghosts_open = list(last["ghosts"])
    w_full = np.asarray(last["w"], dtype=np.float64)

    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))
    n_hold = CH6_N_HOLD * 2
    n_target = 6
    cloud_kw = dict(
        marker_z_mode="bias", marker_axis_lim=axis_lim, zlabel="b",
        title_left=None, title_right=None,
        **CH6_BELIEF_DENSITY_KW,
    )
    smooth = _g("ch3_knob_smoothstep")
    grad_span = 0.052
    ref_s, ref_e, ref_y = _ch6_d1_balanced_six()

    def _duo_layout_frame(**kw):
        base = dict(
            markers=W_orig,
            highlight_w=w_full,
            view_azim=base_azim,
            marker_axis_lim=axis_lim,
            xlim=xlim, ylim=ylim,
            base_study=cs0, base_exam=ce0, base_y=cy0,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
            pop_alpha=0.08,
            w_live=w_full, knob_w=w_full,
            ghost_ws=ghosts_open, ghost_fade_u=0.0,
            **CH6_BELIEF_DENSITY_KW,
        )
        base.update(kw)
        return _finish(_frame_ch6_duo_cloud_layout(**base), clip_id)

    def emit_cloud(W, azim, *, ghost_fade_u=1.0):
        return _finish(
            _ch6_cloud_frame_from_pack(
                pack, markers=W, view_azim=azim, ghost_fade_u=ghost_fade_u,
                **CH6_BELIEF_DENSITY_KW,
            ),
            clip_id,
        )

    def emit_fs(**kw):
        kw.setdefault("grad_span_frac", grad_span)
        kw.setdefault("grad_compact", True)
        return _finish(_frame_ch6_fullscreen_2d_grad(xlim=xlim, ylim=ylim, **kw), clip_id)

    def grad_kw(*, highlight=None):
        return dict(
            show_point_grads_away_from_line=True,
            point_grad_highlight=highlight,
        )

    frames: list = []

    # --- Phase 1–2: duo opening (cloud + fade reel ghosts) ---
    frames.extend(_hold(emit_cloud(W_orig, base_azim), n_hold))
    for u in np.linspace(1.0, 0.0, _draft_short(14, 6)):
        frames.append(emit_cloud(W_orig, base_azim, ghost_fade_u=float(u)))
    frames.extend(_hold(emit_cloud(W_orig, base_azim, ghost_fade_u=0.0), n_hold))

    # --- Fade 3D cloud, then knobs, then resize to fullscreen ---
    n_fade_cloud = _draft_short(14, 6)
    for u in np.linspace(1.0, 0.0, n_fade_cloud):
        frames.append(_duo_layout_frame(cloud_u=float(u), knob_u=1.0, layout_u=0.0))
    frames.extend(_hold(_duo_layout_frame(cloud_u=0.0, knob_u=1.0, layout_u=0.0), n_hold // 2))

    n_fade_knobs = _draft_short(12, 5)
    for u in np.linspace(1.0, 0.0, n_fade_knobs):
        frames.append(_duo_layout_frame(cloud_u=0.0, knob_u=float(u), layout_u=0.0))
    frames.extend(_hold(_duo_layout_frame(cloud_u=0.0, knob_u=0.0, layout_u=0.0), n_hold // 2))

    n_resize = _draft_short(14, 6)
    for u in np.linspace(0.0, 1.0, n_resize):
        frames.append(_duo_layout_frame(
            cloud_u=0.0, knob_u=0.0, layout_u=float(u),
            pop_alpha=0.08 * (1.0 - float(u)),
            pop_study=pop_s if float(u) < 0.98 else None,
            pop_exam=pop_e if float(u) < 0.98 else None,
            pop_y=pop_y if float(u) < 0.98 else None,
        ))
    frames.extend(_hold(emit_fs(
        base_study=cs0, base_exam=ce0, base_y=cy0, w_live=w_full,
    ), n_hold))

    # --- Peel to six (fullscreen, line only — no cloud) ---
    remove_order, _ = _ch6_loo_n6_peel_setup(cs0, ce0, cy0, n_target=n_target)
    visible = np.ones(len(cs0), dtype=bool)
    for rem_i in remove_order:
        if int(visible.sum()) <= n_target:
            break
        visible[rem_i] = False
        if int(visible.sum()) == n_target:
            cs, ce, cy = ref_s, ref_e, ref_y
        else:
            cs = cs0[visible]
            ce = ce0[visible]
            cy = cy0[visible]
        w_live, _ = ch6_fit_dataset(cs, ce, cy)
        frames.append(emit_fs(
            base_study=cs, base_exam=ce, base_y=cy, w_live=w_live,
        ))
    frames.extend(_hold(frames[-1], n_hold))

    s6 = np.asarray(ref_s, dtype=np.float64)
    e6 = np.asarray(ref_e, dtype=np.float64)
    y6 = np.asarray(ref_y, dtype=np.int64)
    w6, _ = ch6_fit_dataset(s6, e6, y6)
    w_opt = np.asarray(w6, dtype=np.float64).copy()

    frames.extend(_hold(emit_fs(
        base_study=s6, base_exam=e6, base_y=y6, w_live=None,
    ), n_hold))

    # --- Grad arrows + spring + outlier (fullscreen) ---
    frames.extend(_hold(emit_fs(
        base_study=s6, base_exam=e6, base_y=y6, w_live=w6, **grad_kw(),
    ), n_hold * 2))

    Xd6 = ch6_design(s6, e6)
    H6 = np.asarray(
        ch6_observed_information(w6, Xd6, y6, ridge=CH6_RIDGE)[:2, :2],
        dtype=np.float64,
    )
    lo1, hi1, lo2, hi2 = CH6_VIEW_BOUNDS_W12
    w_push = w_opt.copy()
    w_push[0] = 0.08
    w_push[1] = float(np.clip(-2.88, lo2, hi2))

    for u in np.linspace(0.0, 1.0, _draft_short(12, 5)):
        ww = (1.0 - float(u)) * w_opt + float(u) * w_push
        frames.append(emit_fs(
            base_study=s6, base_exam=e6, base_y=y6, w_live=ww, **grad_kw(),
        ))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))

    for xy in _hessian_restore_path(
        w_push[:2], w_opt[:2], H6,
        n_frames=_draft_short(24, 10),
        damp=1.15, stiffness=5.0,
    )[1:]:
        ww = w_opt.copy()
        ww[0], ww[1] = float(xy[0]), float(xy[1])
        frames.append(emit_fs(
            base_study=s6, base_exam=e6, base_y=y6, w_live=ww, **grad_kw(),
        ))
    frames.extend(_hold(frames[-1], n_hold))

    s_out, e_out, y_out = 5.0, 2.0, 0
    s7 = np.append(s6, s_out)
    e7 = np.append(e6, e_out)
    y7 = np.append(y6, y_out)
    w7, _ = ch6_fit_dataset(s7, e7, y7)
    outlier_i = len(s6)

    frames.extend(_hold(emit_fs(
        base_study=s7, base_exam=e7, base_y=y7,
        w_live=w_opt, **grad_kw(highlight=outlier_i),
    ), n_hold * 2))

    for u in np.linspace(0.0, 1.0, _draft_short(18, 8)):
        uu = float(u)
        ww = (1.0 - uu) * w_opt + uu * w7
        frames.append(emit_fs(
            base_study=s7, base_exam=e7, base_y=y7,
            w_live=ww, **grad_kw(highlight=outlier_i),
        ))

    frames.extend(_hold(emit_fs(
        base_study=s7, base_exam=e7, base_y=y7,
        w_live=w7, **grad_kw(highlight=outlier_i),
    ), n_hold * 2))

    # --- Drop arrows; morph fullscreen → duo (empty 3D) ---
    frames.extend(_hold(emit_fs(
        base_study=s7, base_exam=e7, base_y=y7, w_live=w7,
    ), n_hold))

    n_layout_back = int(CH5_BEST_LINE_N_LAYOUT)
    for i in range(max(1, n_layout_back)):
        u = smooth(float(i) / max(n_layout_back - 1, 1))
        img = _frame_ch6_wide_to_duo_empty3d(
            u,
            xlim=xlim, ylim=ylim,
            base_study=s6, base_exam=e6, base_y=y6,
            w_live=w6, knob_w=w6, ghost_ws=[], ghost_fade_u=0.0,
            show_3d_u=0.0,
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], n_hold))

    # --- n=6 population dist reel (like ch6_101) with 360° spin ---
    from ch6_frequentist import _CH3_DRAFT

    print(
        f"  ch6_103: building n=6 dist reel ({CH6_DENSITY_END_D1_N_REEL} classrooms)…",
        flush=True,
    )
    pack6 = ch6_population_param_cloud_pack(
        n_target, seed=17, seed_from_key="D1",
        n_reel=CH6_DENSITY_END_D1_N_REEL,
    )
    xlim6, ylim6 = pack6["xlim"], pack6["ylim"]
    pop6_s, pop6_e, pop6_y = pack6["pop_s"], pack6["pop_e"], pack6["pop_y"]
    reel_stride = 10 if not _CH3_DRAFT else 4
    rotate_reel_start_index = 1
    steps6 = pack6["reel_steps"]
    rot_indices = [
        i for i in range(len(steps6))
        if i >= rotate_reel_start_index and (
            i % reel_stride == 0 or i == len(steps6) - 1
        )
    ]
    n_rot = len(rot_indices)
    marker_s6 = ch6_population_n_sweep_marker_size(n_target)
    highlight_s6 = marker_s6 * (70.0 / 28.0)
    for ri, si in enumerate(rot_indices):
        step = steps6[si]
        az = base_azim + 360.0 * float(ri) / max(n_rot - 1, 1)
        img = _frame_duo(
            xlim=xlim6, ylim=ylim6,
            pop_study=pop6_s, pop_exam=pop6_e, pop_y=pop6_y, pop_alpha=0.08,
            base_study=s6, base_exam=e6, base_y=y6,
            w_live=w6, knob_w=w6, highlight_w=w6,
            ghost_ws=step["ghosts"],
            markers=step["markers"],
            view_azim=az,
            marker_s=marker_s6, highlight_marker_s=highlight_s6,
            show_legend=False,
            **cloud_kw,
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], n_hold * 2))

    print(f"  ch6_103: done — {len(frames)} frames", flush=True)
    return frames


def build_ch6_103_cloud_loo_down_n6_spring_grads(clip_id):
    return _build_ch6_103_cloud_loo_down_n6_spring_grads(clip_id)


def _build_ch6_104_population_n7_up_dense_cloud_spin(clip_id):
    """n=7…100 sweep, then histogram/Gaussian projection story on the cloud."""
    from ch6_frequentist import _CH3_DRAFT

    n_reel = 60 if _CH3_DRAFT else int(CH6_DENSITY_END_D1_N_REEL)
    ns = tuple(list(range(7, 52)) + [56, 61, 66, 71, 76, 81, 86, 91, 96, 100])
    if _CH3_DRAFT:
        ns = (7, 8, 9, 12, 20, 40, 70, 100)
    axis_lim, states = _ch6_dense_n_sweep_states_build(
        seed=17, n_reel=n_reel, pad_frac=0.12, ns=ns,
    )
    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))
    n_hold = CH6_N_HOLD * 2
    n_steps = len(states)
    hist_bins = 40 if _CH3_DRAFT else 96
    gauss_grid_n = 64 if _CH3_DRAFT else 180
    hist_bar_pad = 0.18 if _CH3_DRAFT else 0.14
    duo_kw = dict(show_legend=False, title_right=None)
    cloud_kw = dict(
        marker_z_mode="bias", marker_axis_lim=axis_lim, zlabel="b",
        **CH6_BELIEF_DENSITY_KW,
    )

    def _emit_cloud(
        st,
        *,
        azim,
        density_u=0.0,
        projection_u=0.0,
        bins=14,
        gaussian_plane=None,
        gaussian_u=0.0,
        gaussian_callout=None,
        ellipsoid_loops=None,
        ellipsoid_alpha=0.85,
        mu_sigma_u=0.0,
        data_title=None,
    ):
        n_class = int(st["n"])
        marker_s = ch6_population_n_sweep_marker_size(n_class)
        highlight_marker_s = marker_s * (70.0 / 28.0)
        if data_title is None and ellipsoid_loops is None:
            data_title = f"Classroom Size = {n_class}"
        frame_kw = dict(
            xlim=st["xlim"],
            ylim=st["ylim"],
            pop_study=st["pop_s"],
            pop_exam=st["pop_e"],
            pop_y=st["pop_y"],
            pop_alpha=0.08,
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
            marker_s=marker_s,
            highlight_marker_s=highlight_marker_s,
            density_W=st["markers"],
            density_axis_lim=axis_lim,
            density_reveal_u=float(density_u),
            density_projection_u=float(projection_u),
            density_bins=int(bins),
            density_bar_pad=hist_bar_pad,
            gaussian_surface_plane=gaussian_plane,
            gaussian_surface_u=float(gaussian_u),
            gaussian_surface_grid_n=int(gauss_grid_n),
            gaussian_callout=gaussian_callout,
            gaussian_mu_sigma_u=float(mu_sigma_u),
            **cloud_kw,
            **duo_kw,
        )
        if data_title is not None:
            frame_kw["data_title"] = data_title
        if ellipsoid_loops is not None:
            frame_kw.update(
                ellipsoid_loops=ellipsoid_loops,
                **CH6_104_112_HANDOFF_ELLIPSOID_KW,
                ellipsoid_alpha=float(ellipsoid_alpha),
            )
        return _finish(_frame_duo(**frame_kw), clip_id)

    frames: list = []
    # --- Phase 1: grow from n=7 to n=100 (approx Gaussian regime) ---
    for i, st in enumerate(states):
        azim = base_azim + 360.0 * float(i) / max(n_steps - 1, 1)
        frames.append(_emit_cloud(st, azim=azim))
    st = states[-1]
    W = np.asarray(st["markers"], dtype=np.float64)
    stats = ch6_param_stats(W)
    handoff = _ch6_104_112_handoff(stats)
    loops = handoff["loops"]
    lo = float(axis_lim[0])
    mean_pt = handoff["mean_pt"]
    plane_pt = np.array([mean_pt[0], mean_pt[1], lo + 0.03], dtype=np.float64)

    frames.extend(_hold(_emit_cloud(st, azim=base_azim + 360.0), n_hold))

    # --- Phase 2: rotate with small projected histograms ---
    n_rot = _draft_short(72, 24)
    i_pause = max(2, int(0.45 * n_rot))
    for i in range(i_pause):
        az = base_azim + 360.0 + 360.0 * float(i + 1) / max(n_rot, 1)
        frames.append(_emit_cloud(
            st, azim=az, density_u=1.0, projection_u=1.0, bins=hist_bins,
        ))
    pause_az = base_azim + 360.0 + 360.0 * float(i_pause) / max(n_rot, 1)

    # --- Phase 3: pause and reveal Gaussian surface for one projection ---
    n_reveal = _draft_short(20, 8)
    smooth = _g("ch3_knob_smoothstep")
    for j in range(n_reveal):
        u = smooth(float(j) / max(n_reveal - 1, 1))
        frames.append(_emit_cloud(
            st,
            azim=pause_az,
            density_u=1.0,
            projection_u=1.0,
            bins=hist_bins,
            gaussian_plane="wst_el",
            gaussian_u=u,
            gaussian_callout={"point": plane_pt, "label": "gaussian", "color": CH6_VARIANCE_RED},
        ))
    frames.extend(_hold(_emit_cloud(
        st,
        azim=pause_az,
        gaussian_plane="wst_el",
        gaussian_u=1.0,
        gaussian_callout={"point": plane_pt, "label": "gaussian", "color": CH6_VARIANCE_RED},
    ), _draft_short(18, 8)))
    for j in range(n_reveal):
        u = smooth(float(j) / max(n_reveal - 1, 1))
        frames.append(_emit_cloud(
            st,
            azim=pause_az,
            density_u=1.0,
            projection_u=1.0,
            bins=hist_bins,
            gaussian_plane="wst_el",
            gaussian_u=1.0 - u,
            gaussian_callout={"point": plane_pt, "label": "gaussian", "color": CH6_VARIANCE_RED},
        ))

    # --- Phase 4: continue histogram rotation ---
    rem = max(1, n_rot - i_pause)
    for i in range(rem):
        az = pause_az + 360.0 * float(i + 1) / float(rem)
        frames.append(_emit_cloud(
            st, azim=az, density_u=1.0, projection_u=1.0, bins=hist_bins,
        ))

    # --- Phase 5: remove histograms, move arrow to cloud, reveal Gaussian ellipsoids ---
    n_final = _draft_short(28, 10)
    az_phase4_end = pause_az + 360.0
    phase5_spin = _ch6_azim_delta_forward_to_match(az_phase4_end, base_azim)
    for j in range(n_final):
        u = smooth(float(j) / max(n_final - 1, 1))
        call_pt = (1.0 - u) * plane_pt + u * mean_pt
        az = az_phase4_end + phase5_spin * float(j + 1) / max(n_final, 1)
        frames.append(_emit_cloud(
            st,
            azim=az,
            density_u=1.0 - u,
            projection_u=1.0 - u,
            bins=hist_bins,
            ellipsoid_loops=loops,
            ellipsoid_alpha=1.0 * u,
            mu_sigma_u=u,
            data_title=None if u > 0.35 else f"Classroom Size = {len(st['cs'])}",
            gaussian_callout={"point": call_pt, "label": "gaussian", "color": CH6_VARIANCE_RED},
        ))
    handoff_frame = _emit_cloud(
        st,
        azim=base_azim,
        ellipsoid_loops=loops,
        ellipsoid_alpha=1.0,
        mu_sigma_u=1.0,
        data_title=None,
        gaussian_callout=handoff["gaussian_callout"],
    )
    frames.append(handoff_frame)
    frames.extend(_hold(handoff_frame, n_hold))
    print(f"  ch6_104: done — {len(frames)} frames", flush=True)
    return frames


def build_ch6_104_population_n7_up_dense_cloud_spin(clip_id):
    return _build_ch6_104_population_n7_up_dense_cloud_spin(clip_id)


def build_ch6_112_population_n100_zoom_spin_from_104(clip_id):
    """Start at ch6_104 end-state and zoom into the cloud while spinning 360°."""
    from ch6_frequentist import _CH3_DRAFT

    n_reel = 60 if _CH3_DRAFT else int(CH6_DENSITY_END_D1_N_REEL)
    ns = tuple(list(range(7, 52)) + [56, 61, 66, 71, 76, 81, 86, 91, 96, 100])
    if _CH3_DRAFT:
        ns = (7, 8, 9, 12, 20, 40, 70, 100)
    axis_lim, states = _ch6_dense_n_sweep_states_build(
        seed=17, n_reel=n_reel, pad_frac=0.12, ns=ns,
    )
    st = states[-1]
    W = np.asarray(st["markers"], dtype=np.float64)
    stats = ch6_param_stats(W)
    handoff = _ch6_104_112_handoff(stats)
    loops = handoff["loops"]
    mean_pt = handoff["mean_pt"]

    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))
    n_hold = CH6_N_HOLD * 2
    n_spin = _draft_short(84, 28)
    smooth = _g("ch3_knob_smoothstep")

    sampling_ellipsoid = {
        "mean": np.asarray(stats["mean"], dtype=np.float64),
        "cov": np.asarray(stats["cov"], dtype=np.float64),
        "mass": 0.95,
    }

    def _emit(azim, camera_zoom, *, mu_sigma_u=1.0, data_alpha=1.0):
        n_class = int(st["n"])
        marker_s = ch6_population_n_sweep_marker_size(n_class)
        highlight_marker_s = marker_s * (70.0 / 28.0)
        data_alpha = float(np.clip(data_alpha, 0.0, 1.0))
        return _finish(_frame_duo(
            xlim=st["xlim"],
            ylim=st["ylim"],
            pop_study=st["pop_s"],
            pop_exam=st["pop_e"],
            pop_y=st["pop_y"],
            pop_alpha=0.08 * data_alpha,
            base_study=st["cs"],
            base_exam=st["ce"],
            base_y=st["cy"],
            show_base=True,
            ghost_ws=st["ghosts"],
            ghost_fade_u=data_alpha,
            w_live=st["w"],
            knob_w=st["w"],
            markers=st["markers"],
            highlight_w=st["w"],
            marker_s=marker_s,
            highlight_marker_s=highlight_marker_s,
            marker_alpha_scale=data_alpha,
            view_azim=float(azim),
            camera_zoom=float(camera_zoom),
            panel_zoom=float(camera_zoom),
            marker_z_mode="bias",
            marker_axis_lim=axis_lim,
            zlabel="b",
            ellipsoid_loops=loops,
            gaussian_mu_sigma_u=float(mu_sigma_u),
            gaussian_callout=handoff["gaussian_callout"],
            sampling_ellipsoid=sampling_ellipsoid,
            sampling_ellipsoid_reveal_u=0.0,
            show_legend=False,
            title_left=None,
            title_right=None,
            ellipsoid_alpha=1.0,
            **CH6_BELIEF_DENSITY_KW,
            **CH6_104_112_HANDOFF_ELLIPSOID_KW,
        ), clip_id)

    frames: list = []
    frames.extend(_hold(_emit(base_azim, 1.0, mu_sigma_u=1.0, data_alpha=1.0), n_hold))
    peak_zoom = 2.65 if not _CH3_DRAFT else 1.9
    for i in range(1, n_spin + 1):
        t = float(i) / float(max(n_spin, 1))
        u = smooth(t)
        az = base_azim + 360.0 * t
        # True zoom-in then zoom-out while rotating.
        if t <= 0.5:
            zu = smooth(t / 0.5)
        else:
            zu = smooth((1.0 - t) / 0.5)
        zf = 1.0 + (peak_zoom - 1.0) * zu
        data_alpha = max(0.0, 1.0 - 1.05 * zu)
        frames.append(_emit(az, zf, mu_sigma_u=1.0, data_alpha=data_alpha))
    frames.extend(_hold(frames[-1], n_hold))
    return frames


def build_ch6_113_population_n100_zoom_spin_from_104_voxel(clip_id):
    """Variant of ch6_112 using voxel/shell Gaussian ellipsoid rendering."""
    from ch6_frequentist import _CH3_DRAFT

    n_reel = 60 if _CH3_DRAFT else int(CH6_DENSITY_END_D1_N_REEL)
    ns = tuple(list(range(7, 52)) + [56, 61, 66, 71, 76, 81, 86, 91, 96, 100])
    if _CH3_DRAFT:
        ns = (7, 8, 9, 12, 20, 40, 70, 100)
    axis_lim, states = _ch6_dense_n_sweep_states_build(
        seed=17, n_reel=n_reel, pad_frac=0.12, ns=ns,
    )
    st = states[-1]
    W = np.asarray(st["markers"], dtype=np.float64)
    stats = ch6_param_stats(W)
    mean_pt = np.asarray(stats["mean"], dtype=np.float64)
    sampling_ellipsoid = {
        "mean": np.asarray(stats["mean"], dtype=np.float64),
        "cov": np.asarray(stats["cov"], dtype=np.float64),
        "mass": 0.95,
    }

    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))
    n_hold = CH6_N_HOLD * 2
    n_spin = _draft_short(84, 28)
    smooth = _g("ch3_knob_smoothstep")

    def _emit(azim, camera_zoom, *, mu_sigma_u=1.0, data_alpha=1.0):
        n_class = int(st["n"])
        marker_s = ch6_population_n_sweep_marker_size(n_class)
        highlight_marker_s = marker_s * (70.0 / 28.0)
        data_alpha = float(np.clip(data_alpha, 0.0, 1.0))
        return _finish(_frame_duo(
            xlim=st["xlim"],
            ylim=st["ylim"],
            pop_study=st["pop_s"],
            pop_exam=st["pop_e"],
            pop_y=st["pop_y"],
            pop_alpha=0.08 * data_alpha,
            base_study=st["cs"],
            base_exam=st["ce"],
            base_y=st["cy"],
            show_base=True,
            ghost_ws=st["ghosts"],
            ghost_fade_u=data_alpha,
            w_live=st["w"],
            knob_w=st["w"],
            markers=st["markers"],
            highlight_w=st["w"],
            marker_s=marker_s,
            highlight_marker_s=highlight_marker_s,
            marker_alpha_scale=data_alpha,
            view_azim=float(azim),
            camera_zoom=float(camera_zoom),
            panel_zoom=float(camera_zoom),
            marker_z_mode="bias",
            marker_axis_lim=axis_lim,
            zlabel="b",
            sampling_ellipsoid=None,
            gaussian_voxel={
                "mean": sampling_ellipsoid["mean"],
                "cov": sampling_ellipsoid["cov"],
                "mass": 0.95,
                "reveal_u": 1.0,
                "n_cells": 24 if not _CH3_DRAFT else 16,
            },
            gaussian_mu_sigma_u=float(mu_sigma_u),
            gaussian_callout={"point": mean_pt, "label": "gaussian", "color": CH6_VARIANCE_RED},
            show_legend=False,
            title_left=None,
            title_right=None,
            **CH6_BELIEF_DENSITY_KW,
        ), clip_id)

    frames: list = []
    frames.extend(_hold(_emit(base_azim, 1.0, mu_sigma_u=1.0, data_alpha=1.0), n_hold))
    peak_zoom = 2.65 if not _CH3_DRAFT else 1.9
    for i in range(1, n_spin + 1):
        t = float(i) / float(max(n_spin, 1))
        if t <= 0.5:
            zu = smooth(t / 0.5)
        else:
            zu = smooth((1.0 - t) / 0.5)
        zf = 1.0 + (peak_zoom - 1.0) * zu
        az = base_azim + 360.0 * t
        data_alpha = max(0.0, 1.0 - 1.05 * zu)
        frames.append(_emit(az, zf, mu_sigma_u=1.0, data_alpha=data_alpha))
    frames.extend(_hold(frames[-1], n_hold))
    return frames


def build_ch6_114_population_n100_marginal_axes_from_112(clip_id):
    """From ch6_112 end: axis zoom, marginal Gaussians on three walls, finale bridges."""
    from ch6_frequentist import _CH3_DRAFT

    n_reel = 60 if _CH3_DRAFT else int(CH6_DENSITY_END_D1_N_REEL)
    ns = tuple(list(range(7, 52)) + [56, 61, 66, 71, 76, 81, 86, 91, 96, 100])
    if _CH3_DRAFT:
        ns = (7, 8, 9, 12, 20, 40, 70, 100)
    axis_lim, states = _ch6_dense_n_sweep_states_build(
        seed=17, n_reel=n_reel, pad_frac=0.12, ns=ns,
    )
    st = states[-1]
    W = np.asarray(st["markers"], dtype=np.float64)
    stats = ch6_param_stats(W)
    handoff = _ch6_104_112_handoff(stats)
    loops = handoff["loops"]
    tight_lim = _ch6_114_tight_axis_lim(W, axis_lim, pad_frac=0.17)
    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))
    n_hold = CH6_N_HOLD * 2
    smooth = _g("ch3_knob_smoothstep")

    def _lerp_lim(a, b, u):
        t = float(np.clip(u, 0.0, 1.0))
        return (
            float(a[0]) + t * (float(b[0]) - float(a[0])),
            float(a[1]) + t * (float(b[1]) - float(a[1])),
        )

    def _emit(
        *,
        axis_lim_use,
        ellipsoid_alpha=0.0,
        marker_alpha=0.22,
        mu_label_u=1.0,
        layers=None,
        finale_u=0.0,
    ):
        n_class = int(st["n"])
        marker_s = ch6_population_n_sweep_marker_size(n_class)
        highlight_marker_s = marker_s * (70.0 / 28.0)
        return _finish(_frame_duo(
            xlim=st["xlim"],
            ylim=st["ylim"],
            pop_study=st["pop_s"],
            pop_exam=st["pop_e"],
            pop_y=st["pop_y"],
            pop_alpha=0.08,
            base_study=st["cs"],
            base_exam=st["ce"],
            base_y=st["cy"],
            show_base=True,
            ghost_ws=st["ghosts"],
            w_live=st["w"],
            knob_w=st["w"],
            markers=st["markers"],
            highlight_w=st["w"],
            marker_s=marker_s,
            highlight_marker_s=highlight_marker_s,
            marker_alpha_scale=float(marker_alpha),
            view_azim=base_azim,
            marker_z_mode="bias",
            marker_axis_lim=axis_lim_use,
            zlabel="b",
            ellipsoid_loops=loops if ellipsoid_alpha > 1e-4 else None,
            ellipsoid_alpha=float(ellipsoid_alpha),
            mu_label_only_u=float(mu_label_u),
            gaussian_mu_sigma_u=0.0,
            gaussian_callout=None,
            ch6_marginal_layers=layers,
            ch6_marginal_finale_u=float(finale_u),
            show_legend=False,
            title_left=None,
            title_right=None,
            **CH6_BELIEF_DENSITY_KW,
            **CH6_104_112_HANDOFF_ELLIPSOID_KW,
        ), clip_id)

    frames: list = []
    # --- Match ch6_112 end (full labels + ellipsoids) ---
    frames.extend(_hold(_finish(_frame_duo(
        xlim=st["xlim"], ylim=st["ylim"],
        pop_study=st["pop_s"], pop_exam=st["pop_e"], pop_y=st["pop_y"],
        pop_alpha=0.08, base_study=st["cs"], base_exam=st["ce"], base_y=st["cy"],
        show_base=True, ghost_ws=st["ghosts"], w_live=st["w"], knob_w=st["w"],
        markers=st["markers"], highlight_w=st["w"],
        marker_s=ch6_population_n_sweep_marker_size(int(st["n"])),
        highlight_marker_s=ch6_population_n_sweep_marker_size(int(st["n"])) * (70.0 / 28.0),
        view_azim=base_azim, marker_z_mode="bias", marker_axis_lim=axis_lim, zlabel="b",
        ellipsoid_loops=loops, gaussian_mu_sigma_u=1.0,
        gaussian_callout=handoff["gaussian_callout"],
        show_legend=False, title_left=None, title_right=None,
        **CH6_BELIEF_DENSITY_KW, **CH6_104_112_HANDOFF_ELLIPSOID_KW,
        ellipsoid_alpha=1.0,
    ), clip_id), max(1, n_hold // 2)))

    # --- Drop σ² label and gaussian callout; keep μ ---
    n_fade = _draft_short(16, 6)
    for j in range(n_fade):
        u = smooth(float(j) / max(n_fade - 1, 1))
        frames.append(_finish(_frame_duo(
            xlim=st["xlim"], ylim=st["ylim"],
            pop_study=st["pop_s"], pop_exam=st["pop_e"], pop_y=st["pop_y"],
            pop_alpha=0.08, base_study=st["cs"], base_exam=st["ce"], base_y=st["cy"],
            show_base=True, ghost_ws=st["ghosts"], w_live=st["w"], knob_w=st["w"],
            markers=st["markers"], highlight_w=st["w"],
            marker_s=ch6_population_n_sweep_marker_size(int(st["n"])),
            highlight_marker_s=ch6_population_n_sweep_marker_size(int(st["n"])) * (70.0 / 28.0),
            view_azim=base_azim, marker_z_mode="bias", marker_axis_lim=axis_lim, zlabel="b",
            ellipsoid_loops=loops, ellipsoid_alpha=1.0,
            mu_label_only_u=u, gaussian_mu_sigma_u=max(0.0, 1.0 - u),
            gaussian_callout=handoff["gaussian_callout"] if u < 0.5 else None,
            show_legend=False, title_left=None, title_right=None,
            **CH6_BELIEF_DENSITY_KW, **CH6_104_112_HANDOFF_ELLIPSOID_KW,
        ), clip_id))

    # --- Axis zoom (tighten limits, fade ellipsoids) ---
    n_zoom = _draft_short(24, 10)
    for j in range(n_zoom):
        u = smooth(float(j) / max(n_zoom - 1, 1))
        frames.append(_emit(
            axis_lim_use=_lerp_lim(axis_lim, tight_lim, u),
            ellipsoid_alpha=1.0 - u,
            marker_alpha=0.55 - 0.33 * u,
            mu_label_u=1.0,
        ))
    frames.extend(_hold(_emit(
        axis_lim_use=tight_lim, ellipsoid_alpha=0.0, marker_alpha=0.22, mu_label_u=1.0,
    ), max(1, n_hold // 2)))

    completed: list[dict] = []
    n_proj = _draft_short(18, 8)
    n_curve = _draft_short(28, 12)
    n_mean = _draft_short(16, 8)
    n_pass_hold = _draft_short(14, 6)

    for mi, spec in enumerate(CH6_114_MARGINAL_SPECS):
        name, pdf_ax, disp_ax, height_ax, wall_pin, wall_at_hi, pin_map, height_from_hi = spec
        hs = _ch6_114_marginal_height_scale(
            W, height_axis=height_ax, axis_lim=tight_lim, height_from_hi=height_from_hi,
        )
        layer = {
            "spec": spec,
            "height_scale": hs,
            "proj_u": 0.0,
            "curve_u": 0.0,
            "mean_line_u": 0.0,
            "alpha": 1.0,
        }

        for j in range(n_proj):
            u = smooth(float(j) / max(n_proj - 1, 1))
            layer["proj_u"] = u
            frames.append(_emit(
                axis_lim_use=tight_lim, layers=completed + [dict(layer)],
            ))

        layer["proj_u"] = 1.0
        for j in range(n_curve):
            u = smooth(float(j) / max(n_curve - 1, 1))
            layer["curve_u"] = u
            frames.append(_emit(
                axis_lim_use=tight_lim, layers=completed + [dict(layer)],
            ))

        layer["curve_u"] = 1.0
        for j in range(n_mean):
            u = smooth(float(j) / max(n_mean - 1, 1))
            layer["mean_line_u"] = u
            frames.append(_emit(
                axis_lim_use=tight_lim, layers=completed + [dict(layer)],
            ))

        layer["mean_line_u"] = 1.0
        completed.append(dict(layer))
        frames.extend(_hold(_emit(
            axis_lim_use=tight_lim, layers=list(completed),
        ), n_pass_hold))

    # --- Finale: extend wall lines; orthogonal stubs meet at μ ---
    n_fin = _draft_short(26, 10)
    for j in range(n_fin):
        u = smooth(float(j) / max(n_fin - 1, 1))
        frames.append(_emit(
            axis_lim_use=tight_lim,
            layers=completed,
            finale_u=u,
            marker_alpha=0.12,
        ))
    frames.extend(_hold(frames[-1], n_hold))
    return frames


def _ch6_114_end_pack():
    """End-of-ch6_114 state: n=100 cloud, tight axes, completed marginals."""
    from ch6_frequentist import _CH3_DRAFT

    n_reel = 60 if _CH3_DRAFT else int(CH6_DENSITY_END_D1_N_REEL)
    ns = tuple(list(range(7, 52)) + [56, 61, 66, 71, 76, 81, 86, 91, 96, 100])
    if _CH3_DRAFT:
        ns = (7, 8, 9, 12, 20, 40, 70, 100)
    axis_lim, states = _ch6_dense_n_sweep_states_build(
        seed=17, n_reel=n_reel, pad_frac=0.12, ns=ns,
    )
    st = states[-1]
    W = np.asarray(st["markers"], dtype=np.float64)
    stats = ch6_param_stats(W)
    tight_lim = _ch6_114_tight_axis_lim(W, axis_lim, pad_frac=0.17)
    completed: list[dict] = []
    for spec in CH6_114_MARGINAL_SPECS:
        hs = _ch6_114_marginal_height_scale(
            W, height_axis=spec[3], axis_lim=tight_lim, height_from_hi=spec[7],
        )
        completed.append({
            "spec": spec,
            "height_scale": hs,
            "proj_u": 1.0,
            "curve_u": 1.0,
            "mean_line_u": 1.0,
            "alpha": 1.0,
        })
    return {
        "st": st,
        "W": W,
        "stats": stats,
        "mu": np.asarray(stats["mean"], dtype=np.float64).reshape(3),
        "std": np.asarray(stats["std"], dtype=np.float64).reshape(3),
        "tight_lim": tight_lim,
        "base_azim": float(_g("CH3_LIK_W12_CT_AZIM")),
        "completed_layers": completed,
        "xlim": st["xlim"],
        "ylim": st["ylim"],
    }


def _ch6_114_post_marginal_intro_frames(clip_id, pack):
    """Ghost fade, 2D μ line, marginal Gaussian fade — from ch6_114 finale."""
    st = pack["st"]
    mu = pack["mu"]
    tight_lim = pack["tight_lim"]
    smooth = _g("ch3_knob_smoothstep")
    n_hold = max(1, CH6_N_HOLD // 2)
    n_class = int(st["n"])
    marker_s = ch6_population_n_sweep_marker_size(n_class)
    highlight_marker_s = marker_s * (70.0 / 28.0)
    ghost_lo = 0.28

    def _emit(**kw):
        base = dict(
            xlim=pack["xlim"],
            ylim=pack["ylim"],
            pop_study=st["pop_s"],
            pop_exam=st["pop_e"],
            pop_y=st["pop_y"],
            pop_alpha=0.08,
            base_study=st["cs"],
            base_exam=st["ce"],
            base_y=st["cy"],
            show_base=True,
            ghost_ws=st["ghosts"],
            w_live=None,
            knob_w=st["w"],
            markers=st["markers"],
            highlight_w=st["w"],
            marker_s=marker_s,
            highlight_marker_s=highlight_marker_s,
            marker_alpha_scale=0.12,
            view_azim=pack["base_azim"],
            marker_z_mode="bias",
            marker_axis_lim=tight_lim,
            zlabel="b",
            mu_label_only_u=1.0,
            ch6_marginal_layers=pack["completed_layers"],
            ch6_marginal_finale_u=1.0,
            ch6_marginal_global_alpha=1.0,
            show_legend=False,
            title_left=None,
            title_right=None,
            **CH6_BELIEF_DENSITY_KW,
        )
        base.update(kw)
        return _finish(_frame_duo(**base), clip_id)

    frames: list = []
    hold_frame = _emit(ghost_fade_u=1.0, mu_threshold_2d_u=0.0)
    frames.extend(_hold(hold_frame, n_hold))

    n_ghost = _draft_short(18, 8)
    for j in range(n_ghost):
        u = smooth(float(j) / max(n_ghost - 1, 1))
        frames.append(_emit(
            ghost_fade_u=1.0 - u * (1.0 - ghost_lo),
            mu_threshold_2d_u=0.0,
        ))

    n_mu = _draft_short(16, 8)
    for j in range(n_mu):
        u = smooth(float(j) / max(n_mu - 1, 1))
        frames.append(_emit(
            ghost_fade_u=ghost_lo,
            mu_threshold_2d=mu,
            mu_threshold_2d_u=u,
            mu_threshold_2d_lw=3.6,
            mu_threshold_2d_color=CH6_VARIANCE_RED,
        ))

    n_gauss = _draft_short(22, 10)
    for j in range(n_gauss):
        u = smooth(float(j) / max(n_gauss - 1, 1))
        frames.append(_emit(
            ghost_fade_u=ghost_lo,
            mu_threshold_2d=mu,
            mu_threshold_2d_u=1.0,
            ch6_marginal_global_alpha=1.0 - u,
        ))

    frames.extend(_hold(_emit(
        ghost_fade_u=ghost_lo,
        mu_threshold_2d=mu,
        mu_threshold_2d_u=1.0,
        ch6_marginal_layers=None,
        ch6_marginal_finale_u=0.0,
        ch6_marginal_global_alpha=0.0,
    ), n_hold))
    return frames


def _ch6_variance_info_from_114_pack(pack):
    st = pack["st"]
    return {
        "W": pack["W"],
        "mu": pack["mu"],
        "std": pack["std"],
        "xlim": pack["xlim"],
        "ylim": pack["ylim"],
        "cs": st["cs"],
        "ce": st["ce"],
        "cy": st["cy"],
        "w_knob": np.asarray(st["w"], dtype=np.float64).reshape(3),
        "ghost_ws": st["ghosts"],
        "ghost_fade_u": 0.28,
        "marker_axis_lim": pack["tight_lim"],
        "belief_z_lim": pack["tight_lim"],
        "mu_threshold_2d": pack["mu"],
        "mu_threshold_2d_u": 1.0,
        "marker_s": ch6_population_n_sweep_marker_size(int(st["n"])),
    }


def _ch6_append_variance_story_frames(
    frames,
    clip_id,
    info,
    *,
    signed_components=False,
    signed_range_halves=False,
    component_color=None,
    show_box=True,
    box_color=CH6_VARIANCE_POS_COLOR,
    fast=True,
    with_math=False,
    math_from_beat=0,
    positive_contrib=False,
    fade_2d_at_start=False,
    math_end_restore_2d=False,
):
    """Sampling-variance reel (ch6_94-style) on a fixed parameter cloud."""
    W = np.asarray(info["W"], dtype=np.float64)
    mu = np.asarray(info["mu"], dtype=np.float64).reshape(3)
    std = np.asarray(info["std"], dtype=np.float64).reshape(3)
    n_pts = len(W)
    revealed = np.zeros(n_pts, dtype=bool)
    hold = CH6_N_HOLD if fast else CH6_N_HOLD * 2
    init_elev, init_azim = _variance_init_view()
    data_bounds = tuple((float(W[:, i].min()), float(W[:, i].max())) for i in range(3))
    inner_bounds = tuple(
        (float(np.percentile(W[:, i], 8)), float(np.percentile(W[:, i], 92))) for i in range(3)
    )
    accumulated_ranges: dict[int, dict] = {}
    line_color = component_color or CH6_VARIANCE_POS_COLOR
    use_signed_halves = bool(signed_range_halves or signed_components)
    smooth = _g("ch3_knob_smoothstep")
    math_pile_state = {"landed": [], "pile_ymax": None, "beat": None}

    def _header_block(beat_i, axis_idx, *, after_w_el=False):
        if after_w_el:
            return _ch6_var_math_header_block("mean_zero_el")
        if positive_contrib:
            return _ch6_var_math_header_block("variance", axis_idx=int(axis_idx))
        if int(beat_i) == 2 and int(axis_idx) == 1:
            return _ch6_var_math_header_block("deviation_el")
        if int(beat_i) == 1 and int(axis_idx) == 0:
            return _ch6_var_math_header_block("deviation_st")
        return _ch6_var_math_header_block("deviation_st")

    def _mk(beat_i, axis_idx, *, fade=1.0, header=1.0, landed=None, fly=None, pile_ymax=None, after_w_el=False):
        if not with_math or int(beat_i) < int(math_from_beat):
            return {}
        pile_blocks = list(landed) if landed else []
        pile_cap = pile_ymax
        if (
            not fly
            and not pile_blocks
            and pile_cap is None
            and math_pile_state.get("beat") is not None
            and int(math_pile_state["beat"]) == int(beat_i)
            and math_pile_state["landed"]
        ):
            pile_blocks = list(math_pile_state["landed"])
            pile_cap = math_pile_state["pile_ymax"]
        out = dict(
            cov_left_fade_u=float(fade),
            cov_math_header_u=float(header),
            cov_math_header_block=_header_block(beat_i, axis_idx, after_w_el=after_w_el),
            cov_contrib_landed=pile_blocks,
            cov_contrib_fly=fly,
            cov_contrib_ymax=pile_cap,
            cov_positive_only_pile=bool(positive_contrib),
            cov_var_axis=int(axis_idx),
            cov_var_flying=True,
        )
        if after_w_el and float(header) > 0.85:
            out["cov_beat_caption"] = CH6_VAR_MEAN_ZERO_CAPTION
            out["cov_beat_caption_u"] = float(header)
        return out

    def _reveal_var_math_demo(beat_i, axis_idx, *, elev, azim, frame_kw_base=None):
        """Synced deviation-line + flying-block reveal on a balanced demo subset."""
        demo_n = 5 * _draft_short(14, 8)
        demo = _ch6_var_pick_demo(
            W, mu, axis_idx, n=demo_n, seed=115 + int(beat_i),
        )
        scale, pile_ymax = _ch6_var_pile_plan(
            W, mu, demo, axis_idx, positive_only=bool(positive_contrib),
        )
        base_kw = dict(frame_kw_base or {})
        base_kw.setdefault("grey_u", 1.0)
        base_kw.setdefault("show_center", True)
        base_kw.setdefault("view_elev", elev)
        base_kw.setdefault("view_azim", azim)
        base_kw.setdefault("range_state", dict(accumulated_ranges))
        frames.extend(_hold(_vf(
            **base_kw,
            **_mk(beat_i, axis_idx, fade=1.0, header=1.0, pile_ymax=pile_ymax),
        ), max(1, hold // 3)))
        n_fly = _draft_short(14, 8)
        landed: list[tuple[float, str, float]] = []
        pos_cum = 0.0
        neg_cum = 0.0
        revealed[:] = False
        for ki, idx in enumerate(demo):
            shown = list(demo[: ki + 1])
            rev = np.zeros(n_pts, dtype=bool)
            rev[shown] = True
            h, col, sign = _ch6_var_contrib_spec(
                W, mu, int(idx), axis_idx, scale=scale, positive_only=bool(positive_contrib),
            )
            for j in range(n_fly):
                t = smooth(float(j) / max(n_fly - 1, 1))
                frames.append(_vf(
                    **base_kw,
                    revealed_mask=rev,
                    axis_components={axis_idx: {"indices": shown, "alpha": 1.0}},
                    sign_labels_u=1.0 if signed_components else 0.0,
                    **_mk(
                        beat_i, axis_idx, fade=1.0, header=1.0, landed=list(landed),
                        fly=(idx, h, col, sign, t, pos_cum, neg_cum, W, mu),
                        pile_ymax=pile_ymax,
                    ),
                ))
            if positive_contrib:
                pos_cum += float(h)
            elif float(sign) > 0.0:
                pos_cum += float(h)
            else:
                neg_cum -= float(h)
            landed.append((h, col, sign))
            frames.append(_vf(
                **base_kw,
                revealed_mask=rev,
                axis_components={axis_idx: {"indices": shown, "alpha": 1.0}},
                sign_labels_u=1.0 if signed_components else 0.0,
                **_mk(
                    beat_i, axis_idx, fade=1.0, header=1.0, landed=list(landed),
                    pile_ymax=pile_ymax,
                ),
            ))
        frames.extend(_hold(_vf(
            **base_kw,
            revealed_mask=np.isin(np.arange(n_pts), demo),
            axis_components={axis_idx: {"indices": list(demo), "alpha": 1.0}},
            sign_labels_u=1.0 if signed_components else 0.0,
            **_mk(
                beat_i, axis_idx, fade=1.0, header=1.0,
                landed=_ch6_var_landed_from_indices(
                    W, mu, demo, axis_idx, scale=scale, positive_only=bool(positive_contrib),
                ),
                pile_ymax=pile_ymax,
            ),
        ), max(1, hold // 2)))
        math_pile_state["landed"] = _ch6_var_landed_from_indices(
            W, mu, demo, axis_idx, scale=scale, positive_only=bool(positive_contrib),
        )
        math_pile_state["pile_ymax"] = float(pile_ymax)
        math_pile_state["beat"] = int(beat_i)
        return list(demo)

    def _vf(**kw):
        base = dict(
            xlim=info["xlim"],
            ylim=info["ylim"],
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=W,
            mu=mu,
            axis_lim=info["marker_axis_lim"],
            revealed_mask=revealed,
            knob_w=info["w_knob"],
            ghost_ws=info.get("ghost_ws"),
            ghost_fade_u=float(info.get("ghost_fade_u", 1.0)),
            ghost_density_reveal_u=1.0,
            belief_z_lim=info.get("belief_z_lim"),
            mu_threshold_2d=info.get("mu_threshold_2d"),
            mu_threshold_2d_u=float(info.get("mu_threshold_2d_u", 0.0)),
            mu_threshold_2d_lw=3.6,
            mu_threshold_2d_color=CH6_VARIANCE_RED,
            marker_s=float(info.get("marker_s", 28)),
            component_color=line_color,
            range_color=line_color,
            signed_components=signed_components,
            signed_range_halves=use_signed_halves,
            sign_labels=signed_components,
            box_color=box_color,
            axis_component_dotted_lw=CH6_VARIANCE_DOTTED_LW,
        )
        base.update(kw)
        return _finish(_frame_population_variance_duo(**base), clip_id)

    revealed[:] = True
    frames.extend(_hold(_vf(grey_u=0.0, view_elev=init_elev, view_azim=init_azim), max(1, hold // 2)))

    axis_passes = ((2, "b"), (0, "w_st_face"), (1, "w_el_face"))
    revealed[:] = False
    math_intro_done = False
    if fade_2d_at_start and with_math:
        first_axis = int(axis_passes[0][0])
        for u in np.linspace(0.0, 1.0, _draft_short(18, 9)):
            su = float(smooth(float(u)))
            frames.append(_vf(
                grey_u=1.0, show_center=True, view_elev=init_elev, view_azim=init_azim,
                **_mk(0, first_axis, fade=su, header=min(1.0, su * 1.35)),
            ))
        frames.extend(_hold(_vf(
            grey_u=1.0, show_center=True, view_elev=init_elev, view_azim=init_azim,
            **_mk(0, first_axis, fade=1.0, header=1.0),
        ), max(1, hold // 2)))
        math_intro_done = True
    else:
        frames.extend(_hold(_vf(
            grey_u=1.0, show_center=True, view_elev=init_elev, view_azim=init_azim,
        ), hold))

    prev_view = "init"
    math_ending_done = False

    for beat_i, (axis_idx, view_name) in enumerate(axis_passes):
        flip_view = bool(with_math and view_name in ("w_st_face", "w_el_face"))
        ea_a, aa_a = _variance_resolve_view(prev_view)
        ea_b, aa_b = _variance_resolve_view(view_name, flip_180=flip_view)
        az_delta = abs((float(aa_b) - float(aa_a) + 180.0) % 360.0 - 180.0)
        elev_delta = abs(float(ea_b) - float(ea_a))
        if az_delta > 0.5 or elev_delta > 0.5:
            for u in np.linspace(0.0, 1.0, _draft_short(24, 12)):
                elev, azim = _variance_interp_view(
                    prev_view, view_name, float(u), flip_b=flip_view,
                )
                su = float(smooth(float(u)))
                fade_u = 1.0
                header_u = 1.0
                if with_math and int(beat_i) >= int(math_from_beat):
                    if int(beat_i) == int(math_from_beat) and not math_intro_done:
                        fade_u = su
                        header_u = min(1.0, su * 1.35)
                    else:
                        fade_u = 1.0
                        header_u = 1.0
                frames.append(_vf(
                    grey_u=1.0, show_center=True,
                    range_state=dict(accumulated_ranges),
                    view_elev=elev, view_azim=azim,
                    sign_labels_u=1.0 if signed_components else 0.0,
                    **_mk(beat_i, axis_idx, fade=fade_u, header=header_u),
                ))
        prev_view = view_name
        elev, azim = _variance_resolve_view(view_name, flip_180=flip_view)

        use_math_demo = bool(with_math and int(beat_i) >= int(math_from_beat))
        if use_math_demo:
            demo_idx = _reveal_var_math_demo(beat_i, axis_idx, elev=elev, azim=azim)
            all_idx = list(demo_idx)
            frame_kw = dict(
                grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
                range_state=dict(accumulated_ranges),
                axis_components={axis_idx: {"indices": all_idx, "alpha": 1.0}},
                sign_labels_u=1.0 if signed_components else 0.0,
            )
        else:
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
                    sign_labels_u=1.0 if signed_components else 0.0,
                    **_mk(beat_i, axis_idx, fade=1.0, header=1.0),
                ))
            revealed[:] = True
            all_idx = list(range(n_pts))
            frame_kw = dict(
                grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
                range_state=dict(accumulated_ranges),
                axis_components={axis_idx: {"indices": all_idx, "alpha": 1.0}},
                sign_labels_u=1.0 if signed_components else 0.0,
            )
            frames.extend(_hold(_vf(
                **frame_kw,
                **_mk(beat_i, axis_idx, fade=1.0, header=1.0),
            ), max(1, hold // 2)))

        for u in np.linspace(1.0, 0.0, _draft_short(14, 7)):
            frames.append(_vf(
                grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
                range_state=dict(accumulated_ranges),
                axis_components={axis_idx: {"indices": all_idx, "alpha": float(u)}},
                sign_labels_u=float(u) if signed_components else 0.0,
                **_mk(beat_i, axis_idx, fade=1.0, header=1.0),
            ))

        half = 2.0 * float(std[axis_idx])
        sigma_lo = float(mu[axis_idx] - half)
        sigma_hi = float(mu[axis_idx] + half)
        data_lo, data_hi = data_bounds[axis_idx]
        cur_rng = {
            axis_idx: {
                "lo": sigma_lo, "hi": sigma_hi, "alpha": 0.0, "labels": False, "lw": 3.0,
                "color": line_color,
            },
        }
        for u in np.linspace(0.0, 1.0, _draft_short(18, 9)):
            cur_rng[axis_idx]["alpha"] = float(u)
            cur_rng[axis_idx]["labels"] = float(u) > 0.55
            merged = {**accumulated_ranges, **cur_rng}
            frames.append(_vf(
                grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
                range_state=merged,
                **_mk(beat_i, axis_idx, fade=1.0, header=1.0),
            ))
        frames.extend(_hold(_vf(
            grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
            range_state={**accumulated_ranges, **cur_rng},
            **_mk(beat_i, axis_idx, fade=1.0, header=1.0),
        ), max(1, hold // 2)))

        extend_rng = {
            axis_idx: {
                "lo": sigma_lo, "hi": sigma_hi, "alpha": 1.0, "labels": True,
                "lw": 2.8, "color": line_color,
            },
        }
        for u in np.linspace(0.0, 1.0, _draft_short(18, 9)):
            lo = sigma_lo + float(u) * (data_lo - sigma_lo)
            hi = sigma_hi + float(u) * (data_hi - sigma_hi)
            extend_rng[axis_idx] = {
                "lo": lo, "hi": hi, "alpha": 1.0, "labels": True, "lw": 2.8,
                "color": line_color,
            }
            merged = {**accumulated_ranges, **extend_rng}
            frames.append(_vf(
                grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
                range_state=merged,
                **_mk(beat_i, axis_idx, fade=1.0, header=1.0),
            ))
        accumulated_ranges[axis_idx] = {
            "lo": data_lo, "hi": data_hi, "alpha": 1.0, "labels": False, "lw": 2.8,
            "color": line_color,
        }
        frames.extend(_hold(_vf(
            grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
            range_state=dict(accumulated_ranges),
            **_mk(beat_i, axis_idx, fade=1.0, header=1.0),
        ), max(1, hold // 2)))

        if with_math and signed_components and int(beat_i) == 2:
            pl = math_pile_state
            for u in np.linspace(0.0, 1.0, _draft_short(16, 8)):
                hu = float(smooth(float(u)))
                mean_mk = _mk(
                    2, 1, fade=1.0, header=hu, after_w_el=True,
                    landed=pl["landed"], pile_ymax=pl["pile_ymax"],
                )
                frames.append(_vf(
                    grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
                    range_state=dict(accumulated_ranges),
                    **mean_mk,
                ))
            mean_mk = _mk(
                2, 1, fade=1.0, header=1.0, after_w_el=True,
                landed=pl["landed"], pile_ymax=pl["pile_ymax"],
            )
            frames.extend(_hold(_vf(
                grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
                range_state=dict(accumulated_ranges),
                **mean_mk,
            ), hold * 2))

            if math_end_restore_2d:
                for u in np.linspace(1.0, 0.0, _draft_short(22, 11)):
                    su = float(smooth(float(u)))
                    frames.append(_vf(
                        grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
                        range_state=dict(accumulated_ranges),
                        cov_left_fade_u=float(su),
                        cov_math_header_u=float(su),
                        cov_math_header_block=_ch6_var_math_header_block("mean_zero_el"),
                        cov_contrib_landed=list(pl["landed"]) if su > 0.12 else [],
                        cov_contrib_fly=None,
                        cov_contrib_ymax=pl["pile_ymax"] if su > 0.12 else None,
                        cov_positive_only_pile=False,
                        cov_var_axis=1,
                        cov_var_flying=True,
                        cov_beat_caption=CH6_VAR_MEAN_ZERO_CAPTION,
                        cov_beat_caption_u=float(su),
                        mu_threshold_2d_u=float(info.get("mu_threshold_2d_u", 1.0)),
                    ))
                frames.extend(_hold(_vf(
                    grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
                    range_state=dict(accumulated_ranges),
                    mu_threshold_2d_u=float(info.get("mu_threshold_2d_u", 1.0)),
                ), hold))
                math_ending_done = True
                break

        if view_name != axis_passes[-1][1]:
            revealed[:] = False
            frames.extend(_hold(_vf(
                grey_u=1.0, show_center=True, view_elev=elev, view_azim=azim,
                range_state=dict(accumulated_ranges),
                **_mk(beat_i, axis_idx, fade=1.0, header=1.0),
            ), max(1, hold // 3)))

    if math_ending_done:
        return frames

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

    if show_box:
        box_mk = _mk(2, 1, fade=1.0, header=1.0) if with_math else {}
        for u in np.linspace(0.0, 1.0, _draft_short(20, 10)):
            frames.append(_vf(
                grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
                range_state=full_ranges,
                show_box=True, box_bounds=data_bounds, inner_bounds=inner_bounds, gap_u=float(u),
                **box_mk,
            ))
        frames.extend(_hold(_vf(
            grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
            range_state=full_ranges,
            show_box=True, box_bounds=data_bounds, inner_bounds=inner_bounds, gap_u=1.0,
            **box_mk,
        ), hold * 2))

    n_spin = _draft_short(72, 28)
    for t in range(n_spin + 1):
        spin_az = float(azim) + 360.0 * float(t) / float(n_spin)
        spin_kw = dict(
            grey_u=0.0, show_center=True, view_elev=elev, view_azim=spin_az,
            range_state=full_ranges,
        )
        if show_box:
            spin_kw.update(
                show_box=True, box_bounds=data_bounds, inner_bounds=inner_bounds, gap_u=1.0,
            )
        frames.append(_vf(**spin_kw))
    frames.extend(_hold(frames[-1], hold * 2))
    return frames


def build_ch6_115_population_n100_variance_from_114(clip_id):
    """From ch6_114 end: intro handoff, then ch6_94-style variance (cyan lines)."""
    pack = _ch6_114_end_pack()
    frames = _ch6_114_post_marginal_intro_frames(clip_id, pack)
    info = _ch6_variance_info_from_114_pack(pack)
    _ch6_append_variance_story_frames(
        frames, clip_id, info,
        signed_components=False,
        component_color=CH6_VARIANCE_POS_COLOR,
        show_box=True,
        box_color=CH6_VARIANCE_POS_COLOR,
        fast=False,
        with_math=True,
        math_from_beat=0,
        positive_contrib=True,
        fade_2d_at_start=True,
    )
    return frames


def build_ch6_116_population_n100_variance_signed_from_114(clip_id):
    """Variance reel from ch6_114 zoom with +/- colored axis components."""
    pack = _ch6_114_end_pack()
    info = _ch6_variance_info_from_114_pack(pack)
    frames: list = []
    n_hold = max(1, CH6_N_HOLD // 2)

    def _hold_post_intro():
        return _finish(_frame_population_variance_duo(
            xlim=info["xlim"],
            ylim=info["ylim"],
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=info["W"],
            mu=info["mu"],
            axis_lim=info["marker_axis_lim"],
            revealed_mask=np.ones(len(info["W"]), dtype=bool),
            knob_w=info["w_knob"],
            ghost_ws=info.get("ghost_ws"),
            ghost_fade_u=float(info.get("ghost_fade_u", 0.12)),
            ghost_density_reveal_u=1.0,
            belief_z_lim=info.get("belief_z_lim"),
            mu_threshold_2d=info.get("mu_threshold_2d"),
            mu_threshold_2d_u=float(info.get("mu_threshold_2d_u", 1.0)),
            mu_threshold_2d_lw=3.6,
            marker_s=float(info.get("marker_s", 28)),
            grey_u=0.0,
            view_elev=_variance_init_view()[0],
            view_azim=_variance_init_view()[1],
        ), clip_id)

    frames.extend(_hold(_hold_post_intro(), n_hold))
    _ch6_append_variance_story_frames(
        frames, clip_id, info,
        signed_components=True,
        signed_range_halves=True,
        show_box=False,
        fast=False,
        with_math=True,
        math_from_beat=1,
        positive_contrib=False,
        math_end_restore_2d=True,
    )
    return frames


def build_ch6_117_population_n100_box_warp_from_115(clip_id):
    """From ch6_115 end: fade cloud, keep blue box, warp shapes inside the box."""
    pack = _ch6_114_end_pack()
    info = _ch6_variance_info_from_114_pack(pack)
    W_orig = np.asarray(info["W"], dtype=np.float64)
    mu = info["mu"]
    axis_lim = info["marker_axis_lim"]
    full_ranges, box_bounds, inner_bounds = _ch6_variance_end_ranges(W_orig)
    targets = _ch6_warp_target_clouds_in_box(W_orig, box_bounds, axis_lim, seed=117)
    path = (
        targets["original"],
        targets["isotropic"],
        targets["bimodal"],
        targets["elongated"],
        targets["ring"],
        targets["original"],
    )

    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    init_elev, init_azim = _variance_init_view()
    end_azim = float(init_azim) + 360.0
    n_fade = _draft_short(24, 10)
    n_warp = _draft_short(22, 10)
    n_freeze = _draft_short(36, 14)
    line_color = CH6_VARIANCE_POS_COLOR
    revealed = np.ones(len(W_orig), dtype=bool)

    def _emit(W, azim, *, marker_alpha=1.0, ghost_fade_u=None):
        return _finish(_frame_population_variance_duo(
            xlim=info["xlim"],
            ylim=info["ylim"],
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=np.asarray(W, dtype=np.float64),
            mu=mu,
            axis_lim=axis_lim,
            revealed_mask=revealed,
            grey_u=0.0,
            show_center=True,
            range_state=full_ranges,
            range_color=line_color,
            show_box=True,
            box_bounds=box_bounds,
            inner_bounds=inner_bounds,
            gap_u=1.0,
            box_color=line_color,
            view_elev=init_elev,
            view_azim=float(azim),
            knob_w=info["w_knob"],
            ghost_ws=info.get("ghost_ws"),
            ghost_fade_u=float(info.get("ghost_fade_u", 0.28) if ghost_fade_u is None else ghost_fade_u),
            ghost_density_reveal_u=1.0,
            belief_z_lim=info.get("belief_z_lim"),
            mu_threshold_2d=info.get("mu_threshold_2d"),
            mu_threshold_2d_u=float(info.get("mu_threshold_2d_u", 1.0)),
            mu_threshold_2d_lw=3.6,
            marker_s=float(info.get("marker_s", 28)),
            marker_alpha_scale=float(marker_alpha),
        ), clip_id)

    frames: list = []
    frames.extend(_hold(_emit(W_orig, end_azim), n_hold))

    for j in range(n_fade):
        u = smooth(float(j) / max(n_fade - 1, 1))
        frames.append(_emit(
            W_orig, end_azim,
            marker_alpha=1.0 - u,
            ghost_fade_u=float(info.get("ghost_fade_u", 0.28)) * (1.0 - u),
        ))

    frames.extend(_hold(_emit(W_orig, end_azim, marker_alpha=0.0, ghost_fade_u=0.0), max(1, n_hold // 2)))

    az_cursor = end_azim
    n_legs = len(path) - 1
    for leg in range(n_legs):
        a = np.asarray(path[leg], dtype=np.float64)
        b = np.asarray(path[leg + 1], dtype=np.float64)
        for j in range(n_warp):
            u = smooth(float(j) / max(n_warp - 1, 1))
            W = (1.0 - u) * a + u * b
            az = az_cursor + 90.0 * float(j + 1) / max(n_warp, 1)
            marker_alpha = u if leg == 0 else 1.0
            frames.append(_emit(W, az, marker_alpha=marker_alpha, ghost_fade_u=0.0))
        az_cursor += 90.0
        for j in range(n_freeze):
            az = az_cursor + 270.0 * float(j + 1) / max(n_freeze, 1)
            frames.append(_emit(b, az, marker_alpha=1.0, ghost_fade_u=0.0))
        az_cursor += 270.0

    frames.extend(_hold(frames[-1], n_hold))
    return frames


def build_ch6_118_population_n100_hessian_plane_arrows_from_117(clip_id):
    """From ch6_117 end: fade box, rotate 90°, orthogonal Hessian eigen-arrows, 360° orbit."""
    pack = _ch6_114_end_pack()
    end = _ch6_117_end_state(pack)
    info = end["info"]
    mu = end["mu"]
    W = end["W"]
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    n_box_fade = _draft_short(18, 8)
    n_rot = _draft_short(28, 12)
    n_arrow = _draft_short(24, 10)
    n_spin = _draft_short(72, 28)
    revealed = np.ones(len(W), dtype=bool)

    Xd = ch6_design(info["cs"], info["ce"])
    H = ch6_observed_information(mu, Xd, info["cy"], ridge=CH6_RIDGE)
    arrow_dirs, arrow_lens = _ch6_hessian_eigen_arrows(
        mu, H, box_bounds=end["box_bounds"],
    )

    def _emit(azim, *, arrow_u=0.0, show_box=True, box_fade_u=1.0):
        return _finish(_frame_population_variance_duo(
            xlim=info["xlim"],
            ylim=info["ylim"],
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=W,
            mu=mu,
            axis_lim=end["axis_lim"],
            revealed_mask=revealed,
            grey_u=0.0,
            show_center=True,
            range_state=end["full_ranges"],
            range_color=end["line_color"],
            show_box=show_box,
            box_bounds=end["box_bounds"],
            inner_bounds=end["inner_bounds"],
            gap_u=1.0,
            box_color=end["line_color"],
            box_fade_u=float(box_fade_u),
            view_elev=end["elev"],
            view_azim=float(azim),
            knob_w=info["w_knob"],
            ghost_fade_u=0.0,
            belief_z_lim=info.get("belief_z_lim"),
            mu_threshold_2d=info.get("mu_threshold_2d"),
            mu_threshold_2d_u=float(info.get("mu_threshold_2d_u", 1.0)),
            mu_threshold_2d_lw=3.6,
            marker_s=float(info.get("marker_s", 28)),
            marker_alpha_scale=1.0,
            hessian_arrow_dirs=arrow_dirs,
            hessian_arrow_lengths=arrow_lens,
            hessian_arrow_reveal_u=float(arrow_u),
        ), clip_id)

    frames: list = []
    az0 = end["azim"]
    frames.extend(_hold(_emit(az0, arrow_u=0.0, show_box=True, box_fade_u=1.0), n_hold))

    for j in range(n_box_fade):
        u = smooth(float(j) / max(n_box_fade - 1, 1))
        frames.append(_emit(az0, arrow_u=0.0, show_box=True, box_fade_u=1.0 - u))

    for j in range(n_rot):
        u = smooth(float(j) / max(n_rot - 1, 1))
        frames.append(_emit(az0 + 90.0 * u, arrow_u=0.0, show_box=False))

    az1 = az0 + 90.0
    for j in range(n_arrow):
        u = smooth(float(j) / max(n_arrow - 1, 1))
        frames.append(_emit(az1, arrow_u=u, show_box=False))

    frames.extend(_hold(_emit(az1, arrow_u=1.0, show_box=False), max(1, n_hold // 2)))

    for t in range(n_spin + 1):
        az = az1 + 360.0 * float(t) / float(n_spin)
        frames.append(_emit(az, arrow_u=1.0, show_box=False))

    frames.extend(_hold(frames[-1], n_hold))
    return frames


def _ch6_cov_math_overlay_kw(
    text_u: float,
    *,
    beat_caption: str | None = None,
    landed=None,
    pile_ymax=None,
):
    """Left-panel covariance math + caption with a single monotonic opacity."""
    u = float(np.clip(text_u, 0.0, 1.0))
    if u <= 1e-6:
        return {}
    kw = dict(
        cov_left_fade_u=u,
        cov_math_header_u=u,
        cov_contrib_landed=list(landed) if landed else [],
        cov_contrib_ymax=pile_ymax,
    )
    if beat_caption:
        kw["cov_beat_caption"] = str(beat_caption)
        kw["cov_beat_caption_u"] = u
    return kw


def _ch6_cov_math_reveal_kw(text_u: float):
    """Left-panel covariance formula reveal (monotonic; no caption/pile)."""
    return _ch6_cov_math_overlay_kw(text_u)


def _ch6_append_covariance_intro_frames(
    frames,
    *,
    smooth,
    n_hold,
    _emit,
    _ch6_cov_w12_ranges,
    _lerp_lim,
    W_base,
    W_pos,
    mu,
    plane_elev,
    plane_azim,
    start_elev,
    start_azim,
    wst_tgt,
    wel_tgt,
    full_st,
    full_el,
    plane_zoom,
    full_rev,
    include_opening_hold: bool = True,
):
    """3D → plane transition; covariance formula reveals once and stays put."""
    n_arrow_fade = _draft_short(14, 6)
    n_proj = _draft_short(32, 16)
    n_cloud_xfade = _draft_short(18, 10)
    n_pan = _draft_short(40, 18)
    n_zoom = _draft_short(32, 14)
    n_signs = _draft_short(14, 6)
    n_reveal = max(n_proj - 1, 1)

    if include_opening_hold:
        frames.extend(_hold(_emit(
            W_base, elev=start_elev, azim=start_azim,
            revealed_mask=full_rev, marker_alpha=1.0, hessian_u=1.0, grey_u=0.0,
            belief_w12=False,
        ), max(1, n_hold // 2)))

    for j in range(n_arrow_fade):
        u = smooth(float(j) / max(n_arrow_fade - 1, 1))
        frames.append(_emit(
            W_base, elev=start_elev, azim=start_azim,
            revealed_mask=full_rev, marker_alpha=1.0,
            hessian_u=1.0 - u, grey_u=0.0, belief_w12=False,
        ))

    W_proj_end = W_base.copy()
    W_proj_end[:, 2] = float(mu[2])
    for j in range(n_proj):
        u = smooth(float(j) / max(n_proj - 1, 1))
        Wp = W_base.copy()
        Wp[:, 2] = (1.0 - u) * Wp[:, 2] + u * float(mu[2])
        text_u = smooth(float(j) / float(n_reveal))
        frames.append(_emit(
            Wp, elev=start_elev, azim=start_azim,
            revealed_mask=full_rev, marker_alpha=1.0, grey_u=0.0,
            belief_w12=True,
            **_ch6_cov_math_reveal_kw(text_u),
        ))

    for j in range(n_cloud_xfade):
        u = smooth(float(j) / max(n_cloud_xfade - 1, 1))
        Ww = (1.0 - u) * W_proj_end + u * W_pos
        frames.append(_emit(
            Ww, elev=start_elev, azim=start_azim,
            revealed_mask=full_rev, marker_alpha=1.0, grey_u=0.0,
            plane_b_value=float(mu[2]), belief_w12=True,
            **_ch6_cov_math_reveal_kw(1.0),
        ))

    for j in range(n_pan):
        u = smooth(float(j) / max(n_pan - 1, 1))
        elev = (1.0 - u) * start_elev + u * plane_elev
        az_delta = (plane_azim - start_azim + 180.0) % 360.0 - 180.0
        azim = start_azim + u * az_delta
        frames.append(_emit(
            W_pos, elev=elev, azim=azim,
            revealed_mask=full_rev, marker_alpha=1.0, grey_u=0.0,
            camera_zoom=1.0, plane_b_value=float(mu[2]), belief_w12=True,
            **_ch6_cov_math_reveal_kw(1.0),
        ))

    for j in range(n_zoom):
        u = smooth(float(j) / max(n_zoom - 1, 1))
        st_lim = _lerp_lim(full_st, wst_tgt, u)
        el_lim = _lerp_lim(full_el, wel_tgt, u)
        zoom = 1.0 + u * (plane_zoom - 1.0)
        frames.append(_emit(
            W_pos, elev=plane_elev, azim=plane_azim,
            revealed_mask=full_rev, marker_alpha=1.0, grey_u=0.0,
            camera_zoom=zoom, plane_b_value=float(mu[2]), belief_w12=True,
            view_w_st_lim=st_lim, view_w_el_lim=el_lim,
            **_ch6_cov_math_reveal_kw(1.0),
        ))

    ranges = _ch6_cov_w12_ranges(W_pos)
    for j in range(n_signs):
        u = smooth(float(j) / max(n_signs - 1, 1))
        frames.append(_emit(
            W_pos, elev=plane_elev, azim=plane_azim,
            revealed_mask=full_rev, marker_alpha=1.0, grey_u=0.0,
            range_state=ranges, signed_range_halves=True,
            cov_view_signs=True, sign_labels_u=u,
            camera_zoom=plane_zoom, plane_b_value=float(mu[2]), belief_w12=True,
            view_w_st_lim=wst_tgt, view_w_el_lim=wel_tgt,
            **_ch6_cov_math_reveal_kw(1.0),
        ))

    frames.extend(_hold(_emit(
        W_pos, elev=plane_elev, azim=plane_azim,
        revealed_mask=full_rev, marker_alpha=1.0, grey_u=0.0,
        range_state=ranges, signed_range_halves=True,
        cov_view_signs=True, sign_labels_u=1.0,
        camera_zoom=plane_zoom, plane_b_value=float(mu[2]), belief_w12=True,
        view_w_st_lim=wst_tgt, view_w_el_lim=wel_tgt,
        **_ch6_cov_math_reveal_kw(1.0),
    ), max(1, n_hold // 2)))


def _ch6_append_covariance_outro_frames(
    frames,
    *,
    smooth,
    n_hold,
    _emit,
    _ch6_cov_interp_cloud,
    _ch6_cov_w12_ranges,
    _lerp_lim,
    W_base,
    W_ind,
    plane_elev,
    plane_azim,
    start_elev,
    start_azim,
    wst_tgt,
    wel_tgt,
    full_st,
    full_el,
    plane_zoom,
    full_rev,
    beat_caption,
    landed,
    pile_ymax,
):
    """Warp back to the 3D view; covariance math fades out monotonically."""
    n_back_warp = _draft_short(40, 20)
    n_unpan = _draft_short(48, 22)
    n_motion = max(n_back_warp + n_unpan - 1, 1)

    for j in range(n_back_warp):
        u_warp = smooth(float(j) / max(n_back_warp - 1, 1))
        progress = float(j) / float(n_motion)
        text_u = 1.0 - smooth(progress)
        Ww = _ch6_cov_interp_cloud(W_ind, W_base, u_warp)
        frames.append(_emit(
            Ww, elev=plane_elev, azim=plane_azim,
            revealed_mask=full_rev, marker_alpha=1.0, grey_u=0.0,
            range_state=_ch6_cov_w12_ranges(Ww),
            signed_range_halves=True,
            cov_view_signs=True, sign_labels_u=1.0,
            camera_zoom=plane_zoom, belief_w12=True,
            view_w_st_lim=wst_tgt, view_w_el_lim=wel_tgt,
            **_ch6_cov_math_overlay_kw(
                text_u,
                beat_caption=beat_caption,
                landed=landed,
                pile_ymax=pile_ymax,
            ),
        ))

    for j in range(n_unpan):
        u_unpan = smooth(float(j) / max(n_unpan - 1, 1))
        progress = float(n_back_warp + j) / float(n_motion)
        text_u = 1.0 - smooth(progress)
        elev = (1.0 - u_unpan) * plane_elev + u_unpan * start_elev
        az_delta = (start_azim - plane_azim + 180.0) % 360.0 - 180.0
        azim = plane_azim + u_unpan * az_delta
        st_lim = _lerp_lim(wst_tgt, full_st, u_unpan)
        el_lim = _lerp_lim(wel_tgt, full_el, u_unpan)
        zoom = plane_zoom + u_unpan * (1.0 - plane_zoom)
        frames.append(_emit(
            W_base, elev=elev, azim=azim,
            revealed_mask=full_rev, marker_alpha=1.0, grey_u=0.0,
            hessian_u=1.0, belief_w12=False,
            camera_zoom=zoom,
            view_w_st_lim=st_lim, view_w_el_lim=el_lim,
            **_ch6_cov_math_overlay_kw(
                text_u,
                beat_caption=beat_caption,
                landed=landed,
                pile_ymax=pile_ymax,
            ),
        ))

    frames.extend(_hold(_emit(
        W_base, elev=start_elev, azim=start_azim,
        revealed_mask=full_rev, marker_alpha=1.0, hessian_u=1.0, grey_u=0.0,
        belief_w12=False,
    ), n_hold))


def _build_ch6_population_n100_covariance_from_118(
    clip_id, *, with_math: bool = False, outro_only: bool = False, intro_only: bool = False,
):
    """From ch6_118 end: three correlation states (neg/pos/zero), then return to 118 view."""
    pack = _ch6_114_end_pack()
    end = _ch6_118_end_state(pack)
    info = end["info"]
    mu = end["mu"]
    W_base = np.asarray(end["W"], dtype=np.float64)
    axis_lim = end["axis_lim"]
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    n_pts = len(W_base)

    W_neg = _ch6_cov_warp_cloud(W_base, mu, "negative")
    W_pos = _ch6_cov_warp_cloud(W_base, mu, "positive")
    W_ind = _ch6_cov_warp_cloud(W_base, mu, "independent")
    demo_neg = _ch6_cov_pick_demo(W_neg, mu, n_each=_draft_short(22, 10), seed=119)
    demo_pos = _ch6_cov_pick_demo(W_pos, mu, n_each=_draft_short(22, 10), seed=120)
    demo_ind = _ch6_cov_pick_demo(W_ind, mu, n_each=_draft_short(22, 10), seed=121)
    belief_z = info.get("belief_z_lim")

    wst_tgt = CH6_COV_WST_VIEW
    wel_tgt = CH6_COV_WEL_VIEW
    lo_full, hi_full = float(axis_lim[0]), float(axis_lim[1])
    full_st = (lo_full, hi_full)
    full_el = (lo_full, hi_full)

    start_elev, start_azim = float(end["elev"]), float(end["azim"])
    plane_elev, plane_azim = CH6_COV_PLANE_ELEV, CH6_COV_PLANE_AZIM
    plane_zoom = float(CH6_COV_PLANE_ZOOM)

    Xd = ch6_design(info["cs"], info["ce"])
    H = ch6_observed_information(mu, Xd, info["cy"], ridge=CH6_RIDGE)
    arrow_dirs, arrow_lens = _ch6_hessian_eigen_arrows(
        mu, H, box_bounds=end["box_bounds"],
    )

    def _lerp_lim(a, b, u):
        u = float(np.clip(u, 0.0, 1.0))
        return (
            float(a[0]) + u * (float(b[0]) - float(a[0])),
            float(a[1]) + u * (float(b[1]) - float(a[1])),
        )

    def _revealed(indices):
        mask = np.zeros(n_pts, dtype=bool)
        for i in indices:
            mask[int(i)] = True
        return mask

    def _axis_comp(indices, *, alpha=1.0):
        idx = [int(i) for i in indices]
        return {
            0: {"indices": idx, "alpha": float(alpha)},
            1: {"indices": idx, "alpha": float(alpha)},
        }

    def _emit(
        W,
        *,
        elev,
        azim,
        revealed_mask=None,
        marker_alpha=1.0,
        axis_components=None,
        range_state=None,
        signed_range_halves=False,
        sign_labels_u=0.0,
        hessian_u=0.0,
        grey_u=0.0,
        camera_zoom=1.0,
        plane_b_value=None,
        view_w_st_lim=None,
        view_w_el_lim=None,
        belief_w12=False,
        marker_bright_mask=None,
        marker_color_override=None,
        marker_s_scale=1.0,
        marker_dim_scale=0.22,
        cov_view_signs=False,
        cov_quadrant_products=None,
        cov_quadrant_products_u=0.0,
        axis_component_solid_lw=CH6_COV_AXIS_SOLID_LW,
        axis_component_dotted_lw=CH6_COV_AXIS_DOTTED_LW,
        cov_left_fade_u=0.0,
        cov_math_header_u=0.0,
        cov_contrib_landed=None,
        cov_contrib_fly=None,
        cov_contrib_ymax=None,
        cov_beat_caption=None,
        cov_beat_caption_u=0.0,
    ):
        if revealed_mask is None:
            revealed_mask = full_rev
        kw = {}
        if plane_b_value is not None:
            kw["plane_b_value"] = float(plane_b_value)
        if marker_bright_mask is not None:
            kw["marker_bright_mask"] = marker_bright_mask
        if marker_color_override is not None:
            kw["marker_color_override"] = marker_color_override
        return _finish(_frame_population_variance_duo(
            xlim=info["xlim"],
            ylim=info["ylim"],
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=W,
            mu=mu,
            axis_lim=axis_lim,
            revealed_mask=revealed_mask,
            grey_u=float(grey_u),
            show_center=True,
            range_state=range_state,
            signed_range_halves=signed_range_halves,
            signed_components=True,
            sign_labels=False,
            sign_labels_u=float(sign_labels_u),
            sign_label_fontsize=54,
            axis_components=axis_components,
            show_box=False,
            view_elev=float(elev),
            view_azim=float(azim),
            camera_zoom=float(camera_zoom),
            knob_w=info["w_knob"],
            ghost_fade_u=0.0,
            belief_z_lim=belief_z,
            belief_w12=bool(belief_w12),
            mu_threshold_2d=info.get("mu_threshold_2d"),
            mu_threshold_2d_u=float(info.get("mu_threshold_2d_u", 1.0)),
            mu_threshold_2d_lw=3.6,
            marker_s=float(info.get("marker_s", 28)),
            marker_alpha_scale=float(marker_alpha),
            marker_s_scale=float(marker_s_scale),
            marker_dim_scale=float(marker_dim_scale),
            axis_component_solid_lw=float(axis_component_solid_lw),
            axis_component_dotted_lw=float(axis_component_dotted_lw),
            hessian_arrow_dirs=arrow_dirs,
            hessian_arrow_lengths=arrow_lens,
            hessian_arrow_reveal_u=float(hessian_u),
            cov_view_signs=bool(cov_view_signs),
            view_w_st_lim=view_w_st_lim,
            view_w_el_lim=view_w_el_lim,
            cov_quadrant_products=cov_quadrant_products,
            cov_quadrant_products_u=float(cov_quadrant_products_u),
            cov_left_fade_u=float(cov_left_fade_u),
            cov_math_header_u=float(cov_math_header_u),
            cov_contrib_landed=cov_contrib_landed,
            cov_contrib_fly=cov_contrib_fly,
            cov_contrib_ymax=cov_contrib_ymax,
            cov_beat_caption=cov_beat_caption,
            cov_beat_caption_u=float(cov_beat_caption_u),
            **kw,
        ), clip_id)

    def _mk(*, fade=0.0, header=0.0, landed=None, fly=None, pile_ymax=None, beat_caption=None):
        if not with_math:
            return {}
        out = dict(
            cov_left_fade_u=float(fade),
            cov_math_header_u=float(header),
            cov_contrib_landed=list(landed) if landed else [],
            cov_contrib_fly=fly,
            cov_contrib_ymax=pile_ymax,
        )
        if beat_caption is not None:
            out["cov_beat_caption"] = str(beat_caption)
            out["cov_beat_caption_u"] = 1.0
        return out

    def _append_contrib_demo(
        W,
        indices,
        *,
        scale,
        pile_ymax,
        highlight_mode=None,
        quadrant_products=None,
        marker_s_scale=CH6_COV_DEMO_MARKER_SCALE,
        beat_caption=None,
    ):
        if not indices:
            return
        n_fly = _draft_short(14, 8)
        landed: list[tuple[float, str, float]] = []
        pos_cum = 0.0
        neg_cum = 0.0
        for ki, idx in enumerate(indices):
            show = indices[: ki + 1]
            h, col, sign = _ch6_cov_contrib_spec(W, mu, idx, scale=scale)
            for j in range(n_fly):
                t = smooth(float(j) / max(n_fly - 1, 1))
                frames.append(_plane(
                    W, show,
                    highlight_mode=highlight_mode,
                    quadrant_products=quadrant_products,
                    marker_s_scale=marker_s_scale,
                    **_mk(fade=1.0, header=1.0, landed=list(landed),
                          fly=(idx, h, col, sign, t, pos_cum, neg_cum, W, mu),
                          pile_ymax=pile_ymax, beat_caption=beat_caption),
                ))
            if sign > 0.0:
                pos_cum += float(h)
            else:
                neg_cum -= float(h)
            landed.append((h, col, sign))
            frames.append(_plane(
                W, show,
                highlight_mode=highlight_mode,
                quadrant_products=quadrant_products,
                marker_s_scale=marker_s_scale,
                **_mk(fade=1.0, header=1.0, landed=list(landed),
                      pile_ymax=pile_ymax, beat_caption=beat_caption),
            ))

    def _plane(
        W,
        comp_indices,
        *,
        st_lim=wst_tgt,
        el_lim=wel_tgt,
        highlight_mode=None,
        quadrant_products=None,
        marker_s_scale=1.0,
        dim_background=True,
        camera_zoom=plane_zoom,
        marker_dim_scale=0.22,
        cov_left_fade_u=0.0,
        cov_math_header_u=0.0,
        cov_contrib_landed=None,
        cov_contrib_fly=None,
        cov_contrib_ymax=None,
        cov_beat_caption=None,
        cov_beat_caption_u=0.0,
    ):
        comp = [int(i) for i in comp_indices] if comp_indices else []
        ac = _axis_comp(comp) if comp else None
        bright = _revealed(comp) if (dim_background and comp) else full_rev
        override = None
        if highlight_mode and comp:
            override = _ch6_cov_marker_override(
                n_pts, mu, W, comp, mode=str(highlight_mode),
            )
        if comp:
            marker_s_scale = max(float(marker_s_scale), CH6_COV_DEMO_MARKER_SCALE)
            marker_dim_scale = 0.08
        return _emit(
            W,
            elev=plane_elev,
            azim=plane_azim,
            revealed_mask=full_rev,
            marker_alpha=1.0,
            grey_u=0.0,
            range_state=_ch6_cov_w12_ranges(W),
            signed_range_halves=True,
            cov_view_signs=True,
            sign_labels_u=1.0,
            axis_components=ac,
            camera_zoom=float(camera_zoom),
            plane_b_value=float(mu[2]),
            view_w_st_lim=st_lim,
            view_w_el_lim=el_lim,
            belief_w12=True,
            marker_bright_mask=bright,
            marker_color_override=override,
            marker_s_scale=float(marker_s_scale),
            marker_dim_scale=float(marker_dim_scale),
            cov_quadrant_products=quadrant_products,
            cov_quadrant_products_u=1.0 if quadrant_products else 0.0,
            cov_left_fade_u=float(cov_left_fade_u),
            cov_math_header_u=float(cov_math_header_u),
            cov_contrib_landed=cov_contrib_landed,
            cov_contrib_fly=cov_contrib_fly,
            cov_contrib_ymax=cov_contrib_ymax,
            cov_beat_caption=cov_beat_caption,
            cov_beat_caption_u=float(cov_beat_caption_u),
        )

    def _warp(W_a, W_b, n_frames, *, frame_fn):
        for j in range(max(int(n_frames), 1)):
            u = smooth(float(j) / max(int(n_frames) - 1, 1))
            Ww = _ch6_cov_interp_cloud(W_a, W_b, u)
            frames.append(frame_fn(Ww, u))

    frames: list = []
    full_rev = np.ones(n_pts, dtype=bool)

    if intro_only:
        _ch6_append_covariance_intro_frames(
            frames,
            smooth=smooth,
            n_hold=n_hold,
            _emit=_emit,
            _ch6_cov_w12_ranges=_ch6_cov_w12_ranges,
            _lerp_lim=_lerp_lim,
            W_base=W_base,
            W_pos=W_pos,
            mu=mu,
            plane_elev=plane_elev,
            plane_azim=plane_azim,
            start_elev=start_elev,
            start_azim=start_azim,
            wst_tgt=wst_tgt,
            wel_tgt=wel_tgt,
            full_st=full_st,
            full_el=full_el,
            plane_zoom=plane_zoom,
            full_rev=full_rev,
            include_opening_hold=True,
        )
        return frames

    if outro_only:
        mix_n = 5 * _draft_short(16, 10)
        mix_demo = _ch6_cov_pick_skewed_demo(
            W_ind, mu, "balanced", n=mix_n, seed=319,
        )
        mix_scale, mix_ymax = _ch6_cov_pile_plan(W_ind, mu, mix_demo)
        cap3 = CH6_COV_BEAT_CAPTIONS[2]
        mix_landed = _ch6_cov_landed_from_indices(
            W_ind, mu, mix_demo, scale=mix_scale,
        )
        frames.extend(_hold(_plane(
            W_ind, mix_demo,
            highlight_mode="product",
            **_ch6_cov_math_overlay_kw(
                1.0,
                beat_caption=cap3,
                landed=mix_landed,
                pile_ymax=mix_ymax,
            ),
        ), n_hold * 2))
        _ch6_append_covariance_outro_frames(
            frames,
            smooth=smooth,
            n_hold=n_hold,
            _emit=_emit,
            _ch6_cov_interp_cloud=_ch6_cov_interp_cloud,
            _ch6_cov_w12_ranges=_ch6_cov_w12_ranges,
            _lerp_lim=_lerp_lim,
            W_base=W_base,
            W_ind=W_ind,
            plane_elev=plane_elev,
            plane_azim=plane_azim,
            start_elev=start_elev,
            start_azim=start_azim,
            wst_tgt=wst_tgt,
            wel_tgt=wel_tgt,
            full_st=full_st,
            full_el=full_el,
            plane_zoom=plane_zoom,
            full_rev=full_rev,
            beat_caption=cap3,
            landed=mix_landed,
            pile_ymax=mix_ymax,
        )
        return frames

    mismatch_idx = demo_pos["pn"] + demo_pos["np"]
    matched_idx = demo_neg["pp"] + demo_neg["nn"]
    mixed_idx = (
        demo_ind["pp"][:5] + demo_ind["nn"][:5]
        + demo_ind["pn"][:5] + demo_ind["np"][:5]
    )

    _ch6_append_covariance_intro_frames(
        frames,
        smooth=smooth,
        n_hold=n_hold,
        _emit=_emit,
        _ch6_cov_w12_ranges=_ch6_cov_w12_ranges,
        _lerp_lim=_lerp_lim,
        W_base=W_base,
        W_pos=W_pos,
        mu=mu,
        plane_elev=plane_elev,
        plane_azim=plane_azim,
        start_elev=start_elev,
        start_azim=start_azim,
        wst_tgt=wst_tgt,
        wel_tgt=wel_tgt,
        full_st=full_st,
        full_el=full_el,
        plane_zoom=plane_zoom,
        full_rev=full_rev,
        include_opening_hold=True,
    )

    # --- Covariance 1: projected cloud as-is (+×−=−, −×+=−) ---
    mis_n = 5 * _draft_short(14, 8)
    mis_demo = _ch6_cov_pick_skewed_demo(
        W_pos, mu, "neg_dominant", n=mis_n, seed=119,
    )
    mis_scale, mis_ymax = _ch6_cov_pile_plan(W_pos, mu, mis_demo)
    cap1 = CH6_COV_BEAT_CAPTIONS[0]
    frames.extend(_hold(_plane(
        W_pos, [],
        quadrant_products=("pn", "np", "pp", "nn"),
        **_mk(fade=1.0, header=1.0, pile_ymax=mis_ymax, beat_caption=cap1),
    ), max(1, n_hold // 3)))
    _append_contrib_demo(
        W_pos, mis_demo,
        scale=mis_scale,
        pile_ymax=mis_ymax,
        highlight_mode="product",
        quadrant_products=("pn", "np", "pp", "nn"),
        beat_caption=cap1,
    )
    frames.extend(_hold(_plane(
        W_pos, mis_demo,
        highlight_mode="product",
        quadrant_products=("pn", "np", "pp", "nn"),
        **_mk(fade=1.0, header=1.0, landed=_ch6_cov_landed_from_indices(
            W_pos, mu, mis_demo, scale=mis_scale,
        ), pile_ymax=mis_ymax, beat_caption=cap1),
    ), n_hold))

    _warp(
        W_pos, W_neg, _draft_short(26, 12),
        frame_fn=lambda Ww, _u: _plane(
            Ww, [], **_mk(fade=1.0, header=1.0),
        ),
    )

    # --- Covariance 2: flipped cloud (+×+=+, −×−=+) ---
    mat_n = 5 * _draft_short(14, 8)
    mat_demo = _ch6_cov_pick_skewed_demo(
        W_neg, mu, "pos_dominant", n=mat_n, seed=219,
    )
    mat_scale, mat_ymax = _ch6_cov_pile_plan(W_neg, mu, mat_demo)
    cap2 = CH6_COV_BEAT_CAPTIONS[1]
    frames.extend(_hold(_plane(
        W_neg, [],
        quadrant_products=("pp", "nn", "pn", "np"),
        **_mk(fade=1.0, header=1.0, pile_ymax=mat_ymax, beat_caption=cap2),
    ), max(1, n_hold // 3)))
    _append_contrib_demo(
        W_neg, mat_demo,
        scale=mat_scale,
        pile_ymax=mat_ymax,
        highlight_mode="product",
        quadrant_products=("pp", "nn", "pn", "np"),
        beat_caption=cap2,
    )
    frames.extend(_hold(_plane(
        W_neg, mat_demo,
        highlight_mode="product",
        quadrant_products=("pp", "nn", "pn", "np"),
        **_mk(fade=1.0, header=1.0, landed=_ch6_cov_landed_from_indices(
            W_neg, mu, mat_demo, scale=mat_scale,
        ), pile_ymax=mat_ymax, beat_caption=cap2),
    ), n_hold))

    _warp(
        W_neg, W_ind, _draft_short(26, 12),
        frame_fn=lambda Ww, _u: _plane(
            Ww, [], **_mk(fade=1.0, header=1.0),
        ),
    )

    # --- Covariance 3: zero-correlation Gaussian blob ---
    mix_n = 5 * _draft_short(16, 10)
    mix_demo = _ch6_cov_pick_skewed_demo(
        W_ind, mu, "balanced", n=mix_n, seed=319,
    )
    mix_scale, mix_ymax = _ch6_cov_pile_plan(W_ind, mu, mix_demo)
    cap3 = CH6_COV_BEAT_CAPTIONS[2]
    frames.extend(_hold(_plane(
        W_ind, [],
        **_mk(fade=1.0, header=1.0, pile_ymax=mix_ymax, beat_caption=cap3),
    ), max(1, n_hold // 3)))
    _append_contrib_demo(
        W_ind, mix_demo,
        scale=mix_scale,
        pile_ymax=mix_ymax,
        highlight_mode="product",
        beat_caption=cap3,
    )
    frames.extend(_hold(_plane(
        W_ind, mix_demo,
        highlight_mode="product",
        **_mk(fade=1.0, header=1.0, landed=_ch6_cov_landed_from_indices(
            W_ind, mu, mix_demo, scale=mix_scale,
        ), pile_ymax=mix_ymax, beat_caption=cap3),
    ), n_hold * 2))

    _ch6_append_covariance_outro_frames(
        frames,
        smooth=smooth,
        n_hold=n_hold,
        _emit=_emit,
        _ch6_cov_interp_cloud=_ch6_cov_interp_cloud,
        _ch6_cov_w12_ranges=_ch6_cov_w12_ranges,
        _lerp_lim=_lerp_lim,
        W_base=W_base,
        W_ind=W_ind,
        plane_elev=plane_elev,
        plane_azim=plane_azim,
        start_elev=start_elev,
        start_azim=start_azim,
        wst_tgt=wst_tgt,
        wel_tgt=wel_tgt,
        full_st=full_st,
        full_el=full_el,
        plane_zoom=plane_zoom,
        full_rev=full_rev,
        beat_caption=cap3,
        landed=_ch6_cov_landed_from_indices(
            W_ind, mu, mix_demo, scale=mix_scale,
        ),
        pile_ymax=mix_ymax,
    )
    return frames


def build_ch6_119_population_n100_covariance_from_118(clip_id):
    return _build_ch6_population_n100_covariance_from_118(clip_id, with_math=False)


def build_ch6_120_population_n100_covariance_math_from_119(clip_id):
    return _build_ch6_population_n100_covariance_from_118(clip_id, with_math=True)


def build_ch6_124_population_n100_covariance_outro_from_120(clip_id):
    """Beat-3 hold through return to 3D: smooth monotonic covariance-math fade-out."""
    return _build_ch6_population_n100_covariance_from_118(
        clip_id, with_math=True, outro_only=True,
    )


def build_ch6_125_population_n100_covariance_intro_from_120(clip_id):
    """Opening 3D→plane transition through first-beat setup: formula reveals once."""
    return _build_ch6_population_n100_covariance_from_118(
        clip_id, with_math=True, intro_only=True,
    )


def build_ch6_122_population_n100_cov_matrix_contrast_from_120(clip_id):
    """Split-screen: diagonal variances vs full Σ, with 3D vector decomposition."""
    pose = _ch6_matrix_contrast_cloud()
    info = pose["info"]
    mu = pose["mu"]
    W = pose["W"]
    C = pose["C"]
    axis_lim = pose["axis_lim"]
    elev, azim = pose["elev"], pose["azim"]
    marker_s = float(info.get("marker_s", 28))
    belief_z = info.get("belief_z_lim")
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    frames: list = []

    def _emit_120_end():
        return _finish(_frame_population_variance_duo(
            xlim=info["xlim"],
            ylim=info["ylim"],
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=pose["W_full"],
            mu=mu,
            axis_lim=axis_lim,
            revealed_mask=np.ones(len(pose["W_full"]), dtype=bool),
            grey_u=0.0,
            show_center=True,
            view_elev=elev,
            view_azim=azim,
            hessian_arrow_reveal_u=1.0,
            knob_w=info.get("w_knob"),
            ghost_fade_u=0.0,
            marker_s=marker_s,
            belief_z_lim=belief_z,
        ), clip_id)

    def _emit_triptych(**kw):
        return _finish(_frame_ch6_cov_matrix_triptych(
            W=W, mu=mu, C=C, axis_lim=axis_lim, elev=elev, azim=azim,
            marker_s=marker_s, belief_z_lim=belief_z, **kw,
        ), clip_id)

    frames.extend(_hold(_emit_120_end(), max(1, n_hold // 2)))

    n_xfade = _draft_short(28, 14)
    for j in range(n_xfade):
        u = smooth(float(j) / max(n_xfade - 1, 1))
        frames.append(_emit_triptych(layout_u=u, panel_u=u))

    frames.extend(_hold(_emit_triptych(layout_u=1.0, panel_u=1.0), max(1, n_hold // 2)))

    var_u = [0.0, 0.0, 0.0]
    cov_u: dict[tuple[int, int], float] = {}
    cov_visible: set[int] = set()
    left_explored: set[int] = set()
    right_explored: set[tuple[int, int]] = set()

    for axis_j in range(3):
        n_var = _draft_short(16, 8)
        for k in range(n_var):
            u = smooth(float(k) / max(n_var - 1, 1))
            vu = list(var_u)
            vu[axis_j] = u
            frames.append(_emit_triptych(
                layout_u=1.0, panel_u=1.0,
                left_highlight_col=axis_j,
                left_explored_cols=set(left_explored),
                right_explored_cells=set(right_explored),
                var_u=tuple(vu),
                cov_u=dict(cov_u),
                cov_visible_cols=set(cov_visible),
            ))
        var_u[axis_j] = 1.0
        left_explored.add(int(axis_j))

        cov_visible.add(int(axis_j))
        frames.extend(_hold(_emit_triptych(
            layout_u=1.0, panel_u=1.0,
            right_highlight_cell=(int(axis_j), int(axis_j)),
            left_explored_cols=set(left_explored),
            right_explored_cells=set(right_explored),
            var_u=tuple(var_u),
            cov_u=dict(cov_u),
            cov_visible_cols=set(cov_visible),
        ), max(1, n_hold // 2)))
        right_explored.add((int(axis_j), int(axis_j)))

        for axis_i in range(3):
            if int(axis_i) == int(axis_j):
                continue
            frames.extend(_hold(_emit_triptych(
                layout_u=1.0, panel_u=1.0,
                right_highlight_cell=(int(axis_i), int(axis_j)),
                left_explored_cols=set(left_explored),
                right_explored_cells=set(right_explored),
                var_u=tuple(var_u),
                cov_u=dict(cov_u),
                cov_visible_cols=set(cov_visible),
            ), max(1, n_hold // 4)))

            n_cov = _draft_short(14, 7)
            for k in range(n_cov):
                u = smooth(float(k) / max(n_cov - 1, 1))
                cu = dict(cov_u)
                cu[(int(axis_i), int(axis_j))] = u
                frames.append(_emit_triptych(
                    layout_u=1.0, panel_u=1.0,
                    right_highlight_cell=(int(axis_i), int(axis_j)),
                    left_explored_cols=set(left_explored),
                    right_explored_cells=set(right_explored),
                    var_u=tuple(var_u),
                    cov_u=cu,
                    cov_visible_cols=set(cov_visible),
                ))
            cov_u[(int(axis_i), int(axis_j))] = 1.0
            right_explored.add((int(axis_i), int(axis_j)))
            frames.extend(_hold(_emit_triptych(
                layout_u=1.0, panel_u=1.0,
                right_highlight_cell=(int(axis_i), int(axis_j)),
                left_explored_cols=set(left_explored),
                right_explored_cells=set(right_explored),
                var_u=tuple(var_u),
                cov_u=dict(cov_u),
                cov_visible_cols=set(cov_visible),
            ), max(1, n_hold // 3)))

    frames.extend(_hold(_emit_triptych(
        layout_u=1.0, panel_u=1.0,
        left_explored_cols=set(left_explored),
        right_explored_cells=set(right_explored),
        var_u=tuple(var_u),
        cov_u=dict(cov_u),
        cov_visible_cols=set(cov_visible),
    ), n_hold * 2))
    return frames


def build_ch6_123_population_n100_cov_matrix_shrink_spin_from_122(clip_id):
    """From ch6_122 end: panels out, Σ column arrows shrink to true scale, 270° spin."""
    pose = _ch6_matrix_contrast_cloud()
    info = pose["info"]
    mu = pose["mu"]
    W = np.asarray(pose["W"], dtype=np.float64)
    C = np.asarray(pose["C"], dtype=np.float64)
    axis_lim = pose["axis_lim"]
    elev, azim0 = pose["elev"], pose["azim"]
    marker_s = float(info.get("marker_s", 28))
    mat = _ch6_122_end_matrix_state()
    mu_line = np.asarray(info.get("w_knob", mu), dtype=np.float64).reshape(3)
    belief_z = info.get("belief_z_lim")
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    frames: list = []

    def _emit_triptych(**kw):
        defaults = dict(
            W=W, mu=mu, C=C, axis_lim=axis_lim, elev=elev, azim=azim0,
            marker_s=marker_s, layout_u=1.0, panel_u=1.0,
            left_explored_cols=mat["left_explored_cols"],
            right_explored_cells=mat["right_explored_cells"],
            var_u=mat["var_u"], cov_u=mat["cov_u"],
            cov_visible_cols=mat["cov_visible_cols"],
            arrow_boost=CH6_MATRIX_ARROW_BOOST,
            draw_variance=False,
            variance_u=0.0,
            eigen_u=0.0,
            belief_z_lim=belief_z,
        )
        defaults.update(kw)
        return _finish(_frame_ch6_cov_matrix_triptych(**defaults), clip_id)

    def _emit_duo(
        azim,
        *,
        arrow_boost=1.0,
        threshold_u=1.0,
    ):
        return _finish(_frame_population_variance_duo(
            xlim=info["xlim"],
            ylim=info["ylim"],
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=W,
            mu=mu,
            axis_lim=axis_lim,
            revealed_mask=np.ones(len(W), dtype=bool),
            grey_u=0.0,
            show_center=True,
            view_elev=elev,
            view_azim=float(azim),
            knob_w=info.get("w_knob"),
            ghost_fade_u=0.0,
            marker_s=marker_s,
            belief_z_lim=belief_z,
            mu_threshold_2d=mu_line,
            mu_threshold_2d_u=float(threshold_u),
            mu_threshold_2d_lw=3.6,
            mu_threshold_2d_color=CH6_VARIANCE_RED,
            matrix_C=C,
            matrix_var_u=mat["var_u"],
            matrix_cov_u=mat["cov_u"],
            matrix_cov_visible_cols=mat["cov_visible_cols"],
            matrix_arrow_boost=float(arrow_boost),
            matrix_draw_variance=False,
            matrix_variance_u=0.0,
            matrix_eigen_u=0.0,
        ), clip_id)

    frames.extend(_hold(_emit_triptych(), n_hold))

    n_morph = _draft_short(44, 22)
    for k in range(n_morph):
        u = smooth(float(k) / max(n_morph - 1, 1))
        layout_u = 1.0 - u
        panel_u = 1.0 - u
        boost = CH6_MATRIX_ARROW_BOOST + u * (1.0 - CH6_MATRIX_ARROW_BOOST)
        duo_u = smooth(max(0.0, (u - 0.55) / 0.45))
        if layout_u > 0.04:
            frames.append(_emit_triptych(
                layout_u=layout_u,
                panel_u=panel_u,
                arrow_boost=boost,
            ))
        else:
            frames.append(_emit_duo(
                azim0,
                arrow_boost=boost,
                threshold_u=duo_u,
            ))

    n_spin = _draft_short(54, 22)
    for k in range(n_spin + 1):
        u = smooth(float(k) / max(n_spin, 1))
        frames.append(_emit_duo(
            azim0 + 270.0 * u,
            arrow_boost=1.0,
            threshold_u=1.0,
        ))

    frames.extend(_hold(_emit_duo(azim0 + 270.0), n_hold))
    return frames


def _build_ch6_nll_voxel_landscape_from_123(clip_id, *, use_d1_transition: bool):
    """From ch6_123 end: fade cloud/Σ, NLL voxel landscape, 360° spin.

    With ``use_d1_transition``, crossfade the 2D roster to D1 and voxelize D1;
    otherwise keep the n=100 classroom on stage and voxelize it.
    """
    pose = _ch6_matrix_contrast_cloud()
    info = pose["info"]
    mu = pose["mu"]
    W = np.asarray(pose["W"], dtype=np.float64)
    C = np.asarray(pose["C"], dtype=np.float64)
    axis_lim = pose["axis_lim"]
    elev0, azim0 = pose["elev"], pose["azim"]
    azim_end = float(azim0) + 270.0
    marker_s = float(info.get("marker_s", 28))
    mat = _ch6_122_end_matrix_state()
    mu_line = np.asarray(info.get("w_knob", mu), dtype=np.float64).reshape(3)
    belief_z = info.get("belief_z_lim")
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2

    class_study = np.asarray(info["cs"], dtype=np.float64)
    class_exam = np.asarray(info["ce"], dtype=np.float64)
    class_y = np.asarray(info["cy"], dtype=np.float64).reshape(-1)
    if use_d1_transition:
        d1_study, d1_exam, d1_y = ch5_unpack_dataset("D1")
        d1_xlim, d1_ylim = ch5_plot_limits("D1")
        voxel_study, voxel_exam, voxel_y = d1_study, d1_exam, d1_y
        panel_xlim, panel_ylim = d1_xlim, d1_ylim
    else:
        d1_study = d1_exam = d1_y = None
        d1_xlim = d1_ylim = None
        voxel_study, voxel_exam, voxel_y = class_study, class_exam, class_y
        panel_xlim, panel_ylim = info["xlim"], info["ylim"]

    ax_lo, ax_hi = float(axis_lim[0]), float(axis_lim[1])
    voxel_bounds = (ax_lo, ax_hi, ax_lo, ax_hi, ax_lo, ax_hi)
    voxel_cache = _ch6_nll_voxel_cache(
        voxel_study, voxel_exam, voxel_y, bounds=voxel_bounds,
    )

    ct_elev = ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV")))
    ct_azim = float(_g("CH3_LIK_W12_CT_AZIM"))

    def _lerp_lim(a, b, u):
        return tuple(float(x) + float(u) * (float(y) - float(x)) for x, y in zip(a, b))

    def _emit(
        *,
        azim,
        elev,
        pop_fade=0.0,
        mat_fade=0.0,
        threshold_u=0.0,
        voxel_u=0.0,
        curv_ref_pct=None,
        show_voxels: bool = False,
        dataset_blend=0.0,
        xlim=None,
        ylim=None,
    ):
        blend = float(np.clip(float(dataset_blend), 0.0, 1.0))
        kw = dict(
            xlim=xlim if xlim is not None else info["xlim"],
            ylim=ylim if ylim is not None else info["ylim"],
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=W,
            mu=mu,
            axis_lim=axis_lim,
            revealed_mask=np.ones(len(W), dtype=bool),
            grey_u=0.0,
            show_center=True,
            view_elev=float(elev),
            view_azim=float(azim),
            knob_w=info.get("w_knob"),
            ghost_fade_u=0.0,
            marker_s=marker_s,
            belief_z_lim=belief_z,
            mu_threshold_2d=mu_line,
            mu_threshold_2d_u=float(threshold_u),
            mu_threshold_2d_lw=3.6,
            mu_threshold_2d_color=CH6_VARIANCE_RED,
            matrix_C=C,
            matrix_var_u=mat["var_u"],
            matrix_cov_u=mat["cov_u"],
            matrix_cov_visible_cols=mat["cov_visible_cols"],
            matrix_arrow_boost=1.0,
            matrix_draw_variance=False,
            matrix_variance_u=0.0,
            matrix_eigen_u=0.0,
            population_fade_u=float(pop_fade),
            matrix_fade_u=float(mat_fade),
            voxel_view_bounds=voxel_bounds,
        )
        if use_d1_transition:
            kw["alt_base_study"] = d1_study
            kw["alt_base_exam"] = d1_exam
            kw["alt_base_y"] = d1_y
            kw["dataset_blend_u"] = blend
        if show_voxels:
            kw["nll_voxel_cache"] = voxel_cache
            kw["nll_voxel_sweep_u"] = float(voxel_u)
            if curv_ref_pct is not None:
                kw["nll_voxel_curv_ref_pct"] = float(curv_ref_pct)
        return _finish(_frame_population_variance_duo(**kw), clip_id)

    frames: list = []
    frames.extend(_hold(_emit(
        azim=azim_end, elev=elev0,
        pop_fade=1.0, mat_fade=1.0, threshold_u=1.0, dataset_blend=0.0,
    ), n_hold))

    n_fade = _draft_short(44 if use_d1_transition else 36, 22 if use_d1_transition else 18)
    for k in range(n_fade):
        u = smooth(float(k) / max(n_fade - 1, 1))
        fade = smooth(u)
        if use_d1_transition:
            xl = _lerp_lim(info["xlim"], d1_xlim, fade)
            yl = _lerp_lim(info["ylim"], d1_ylim, fade)
            blend = fade
        else:
            xl = info["xlim"]
            yl = info["ylim"]
            blend = 0.0
        frames.append(_emit(
            azim=azim_end,
            elev=elev0,
            pop_fade=1.0 - fade,
            mat_fade=1.0 - fade,
            threshold_u=1.0 - fade,
            dataset_blend=blend,
            xlim=xl,
            ylim=yl,
        ))

    frames.extend(_hold(_emit(
        azim=azim_end, elev=elev0,
        pop_fade=0.0, mat_fade=0.0, threshold_u=0.0,
        dataset_blend=1.0 if use_d1_transition else 0.0,
        xlim=panel_xlim, ylim=panel_ylim,
    ), max(1, n_hold // 2)))

    n_sweep = _draft_short(72, 28)
    ref_max = float(CH6_NLL_VOXEL_CURV_REF_PCT_MAX)
    ref_min = float(CH6_NLL_VOXEL_CURV_REF_PCT_MIN)
    for k in range(n_sweep):
        u = smooth(float(k) / max(n_sweep - 1, 1))
        frames.append(_emit(
            azim=ct_azim,
            elev=ct_elev,
            show_voxels=True,
            voxel_u=u,
            curv_ref_pct=ref_max,
            dataset_blend=1.0 if use_d1_transition else 0.0,
            xlim=panel_xlim,
            ylim=panel_ylim,
        ))

    frames.extend(_hold(_emit(
        azim=ct_azim, elev=ct_elev,
        show_voxels=True, voxel_u=1.0,
        curv_ref_pct=ref_max,
        dataset_blend=1.0 if use_d1_transition else 0.0,
        xlim=panel_xlim, ylim=panel_ylim,
    ), max(1, n_hold // 2)))

    n_spin = _draft_short(80, 32)
    for k in range(n_spin + 1):
        u = smooth(float(k) / max(n_spin, 1))
        frames.append(_emit(
            azim=ct_azim + 360.0 * u,
            elev=ct_elev,
            show_voxels=True,
            voxel_u=1.0,
            curv_ref_pct=_ch6_nll_voxel_curv_ref_pct_between(u, ref_max, ref_min),
            dataset_blend=1.0 if use_d1_transition else 0.0,
            xlim=panel_xlim,
            ylim=panel_ylim,
        ))

    for k in range(n_spin + 1):
        u = smooth(float(k) / max(n_spin, 1))
        frames.append(_emit(
            azim=ct_azim + 360.0 + 360.0 * u,
            elev=ct_elev,
            show_voxels=True,
            voxel_u=1.0,
            curv_ref_pct=_ch6_nll_voxel_curv_ref_pct_between(u, ref_min, ref_max),
            dataset_blend=1.0 if use_d1_transition else 0.0,
            xlim=panel_xlim,
            ylim=panel_ylim,
        ))

    frames.extend(_hold(_emit(
        azim=ct_azim + 720.0, elev=ct_elev,
        show_voxels=True, voxel_u=1.0,
        curv_ref_pct=ref_max,
        dataset_blend=1.0 if use_d1_transition else 0.0,
        xlim=panel_xlim, ylim=panel_ylim,
    ), n_hold))
    return frames


def build_ch6_126_population_n100_nll_voxel_landscape_from_123(clip_id):
    """From ch6_123 end: morph classroom → D1, then D1 NLL voxel landscape + 2×360° spin."""
    return _build_ch6_nll_voxel_landscape_from_123(clip_id, use_d1_transition=True)


def build_ch6_127_population_n100_nll_voxel_landscape_classroom_from_123(clip_id):
    """From ch6_123 end: fade cloud/Σ, n=100 classroom NLL voxel landscape + 2×360° spin."""
    return _build_ch6_nll_voxel_landscape_from_123(clip_id, use_d1_transition=False)


def build_ch6_128_population_n100_nll_voxel_ct_scan_from_127(clip_id):
    """From ch6_127 end: reveal CT plane, then Ch4-style NLL axis sweeps on classroom n=100."""
    pose = _ch6_matrix_contrast_cloud()
    info = pose["info"]
    mu = pose["mu"]
    W = np.asarray(pose["W"], dtype=np.float64)
    C = np.asarray(pose["C"], dtype=np.float64)
    axis_lim = pose["axis_lim"]
    marker_s = float(info.get("marker_s", 28))
    mat = _ch6_122_end_matrix_state()
    mu_line = np.asarray(info.get("w_knob", mu), dtype=np.float64).reshape(3)
    belief_z = info.get("belief_z_lim")
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2

    study = np.asarray(info["cs"], dtype=np.float64)
    exam = np.asarray(info["ce"], dtype=np.float64)
    y = np.asarray(info["cy"], dtype=np.float64).reshape(-1)
    panel_xlim, panel_ylim = info["xlim"], info["ylim"]

    ax_lo, ax_hi = float(axis_lim[0]), float(axis_lim[1])
    voxel_bounds = (ax_lo, ax_hi, ax_lo, ax_hi, ax_lo, ax_hi)
    voxel_cache = _ch6_nll_voxel_cache(study, exam, y, bounds=voxel_bounds)

    ct_elev = ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV")))
    ct_azim = float(_g("CH3_LIK_W12_CT_AZIM"))
    azim_end = ct_azim + 720.0
    ref_max = float(CH6_NLL_VOXEL_CURV_REF_PCT_MAX)
    ct_axes = tuple(_g("CH3_LIK_CT_AXES"))
    ct_first_axis = str(ct_axes[0])
    ct_start_lo, _ = _g("_ch4_ct_axis_limits")(ct_first_axis, voxel_bounds)
    ct_start_lo = float(ct_start_lo)

    pivot_map = dict(_g("CH3_LIK_CT_PIVOTS"))
    n_ct_hold = _draft_short(int(_g("CH3_LIK_CT_N_HOLD")), 4)
    n_ct_sweep = _draft_short(int(_g("CH3_LIK_CT_N_SWEEP")), 8)
    n_ct_pivot = _draft_short(int(_g("CH3_LIK_CT_N_PIVOT")), 6)

    def _emit(
        *,
        azim=azim_end,
        elev=ct_elev,
        voxel_u=0.0,
        voxel_alpha_u=0.0,
        curv_ref_pct=ref_max,
        ct_sweep_axis=None,
        ct_plane_val=None,
        ct_pivot_from=None,
        ct_pivot_to=None,
        ct_pivot_u=0.0,
        ct_alpha_u=0.0,
    ):
        kw = dict(
            xlim=panel_xlim,
            ylim=panel_ylim,
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=W,
            mu=mu,
            axis_lim=axis_lim,
            revealed_mask=np.ones(len(W), dtype=bool),
            grey_u=0.0,
            show_center=True,
            view_elev=float(elev),
            view_azim=float(azim),
            knob_w=info.get("w_knob"),
            ghost_fade_u=0.0,
            marker_s=marker_s,
            belief_z_lim=belief_z,
            mu_threshold_2d=mu_line,
            mu_threshold_2d_u=0.0,
            mu_threshold_2d_lw=3.6,
            mu_threshold_2d_color=CH6_VARIANCE_RED,
            matrix_C=C,
            matrix_var_u=mat["var_u"],
            matrix_cov_u=mat["cov_u"],
            matrix_cov_visible_cols=mat["cov_visible_cols"],
            matrix_arrow_boost=1.0,
            matrix_draw_variance=False,
            matrix_variance_u=0.0,
            matrix_eigen_u=0.0,
            population_fade_u=0.0,
            matrix_fade_u=0.0,
            nll_voxel_cache=voxel_cache,
            nll_voxel_sweep_u=float(voxel_u),
            nll_voxel_curv_ref_pct=float(curv_ref_pct),
            nll_voxel_alpha_u=float(voxel_alpha_u),
            voxel_view_bounds=voxel_bounds,
            nll_ct_bounds=voxel_bounds,
            nll_ct_study=study,
            nll_ct_exam=exam,
            nll_ct_y=y,
            nll_ct_sweep_axis=ct_sweep_axis,
            nll_ct_plane_val=ct_plane_val,
            nll_ct_pivot_from=ct_pivot_from,
            nll_ct_pivot_to=ct_pivot_to,
            nll_ct_pivot_u=float(ct_pivot_u),
            nll_ct_alpha_u=float(ct_alpha_u),
        )
        return _finish(_frame_population_variance_duo(**kw), clip_id)

    frames: list = []
    end_kw = dict(
        voxel_u=1.0,
        voxel_alpha_u=1.0,
        curv_ref_pct=ref_max,
    )
    frames.extend(_hold(_emit(**end_kw), max(1, n_hold // 2)))

    n_reveal = _draft_short(52, 20)
    for k in range(n_reveal):
        u = smooth(float(k) / max(n_reveal - 1, 1))
        frames.append(_emit(
            voxel_u=1.0,
            voxel_alpha_u=1.0 - u,
            curv_ref_pct=ref_max,
            ct_sweep_axis=ct_first_axis,
            ct_plane_val=ct_start_lo,
            ct_alpha_u=u,
        ))

    ct_only = dict(
        ct_sweep_axis=ct_first_axis,
        ct_plane_val=ct_start_lo,
        ct_alpha_u=1.0,
    )
    frames.extend(_hold(_emit(**ct_only), max(1, n_hold // 2)))

    for i, axis in enumerate(ct_axes):
        if i > 0:
            prev = ct_axes[i - 1]
            nxt = pivot_map.get(prev)
            if nxt == axis:
                for tv in np.linspace(0.0, 1.0, n_ct_pivot, endpoint=True):
                    frames.append(_emit(
                        ct_pivot_from=prev,
                        ct_pivot_to=axis,
                        ct_pivot_u=float(tv),
                        ct_alpha_u=1.0,
                    ))
        lo, hi = _g("_ch4_ct_axis_limits")(axis, voxel_bounds)
        if i == 0:
            for _ in range(n_ct_hold):
                frames.append(_emit(
                    ct_sweep_axis=axis,
                    ct_plane_val=float(lo),
                    ct_alpha_u=1.0,
                ))
        for tv in np.linspace(0.0, 1.0, n_ct_sweep, endpoint=True):
            u = smooth(float(tv))
            val = float(lo) + u * (float(hi) - float(lo))
            frames.append(_emit(
                ct_sweep_axis=axis,
                ct_plane_val=val,
                ct_alpha_u=1.0,
            ))

    frames.extend(_hold(frames[-1], n_hold))
    return frames


def build_ch6_129_population_n100_classroom_spring_from_127(clip_id):
    """From ch6_127 end: fade voxels → cloud → μ vs classroom → zoom → grads → spring."""
    return _build_ch6_129_population_n100_classroom_spring_from_127(clip_id, voxel_trail=False)


def build_ch6_130_population_n100_classroom_spring_voxel_trail_from_127(clip_id):
    """Same as ch6_129 but accumulates classroom-NLL voxel cubes along the walker path."""
    return _build_ch6_129_population_n100_classroom_spring_from_127(clip_id, voxel_trail=True)


def build_ch6_131_population_n100_classroom_w12_push_from_127(clip_id):
    """Two points → radial NLL reveal → zoom → gradient → low-curvature push."""
    return _build_ch6_131_population_n100_classroom_w12_push(clip_id)


def build_ch6_132_population_n100_classroom_w12_curvature_contrast_from_131(clip_id):
    """Zoomed pair → NLL out → steep bowl → same gradient, shorter push."""
    return _build_ch6_132_population_n100_classroom_w12_curvature_contrast(clip_id)


def _ch6_w12_push_arrow_scale(mu_w12, xy, H, *, curv_mult=1.0):
    """Push-arrow strength decaying with Mahalanobis distance from the average."""
    H_eff = float(curv_mult) * np.asarray(H, dtype=np.float64).reshape(2, 2)
    evals = np.linalg.eigvalsh(H_eff)
    floor = max(0.08, -float(np.min(evals)) + 0.08)
    H_eff = H_eff + floor * np.eye(2, dtype=np.float64)
    d = np.asarray(xy, dtype=np.float64).reshape(2) - mu_w12[:2]
    return float(np.exp(-0.62 * float(d @ H_eff @ d)))


def _ch6_w12_push_emit_factory(
    pack, clip_id, *, alt_surface=None, reveal_origin=None,
):
    """Shared duo-frame emitter for ch6_131 / ch6_132."""
    info = pack["info"]
    mu = pack["mu"]
    W = pack["W"]
    mu_w12 = pack["mu_w12"]
    w_class = pack["w_class"]
    w_knob = pack["w_knob"]
    axis_lim = pack["axis_lim"]
    w12_full = pack["w12_full"]
    class_surf = pack["class_surf"]
    alt_surf = alt_surface
    rev_origin = reveal_origin if reveal_origin is not None else w_class
    belief_z = info.get("belief_z_lim")
    marker_s = float(info.get("marker_s", 28))
    smooth = _g("ch3_knob_smoothstep")
    ct_elev = ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV")))
    azim_end = float(_g("CH3_LIK_W12_CT_AZIM")) + 720.0
    g_scale = CH6_W12_PUSH_GRAD_SCALE

    def _surface_markers():
        return [
            {"point": mu_w12, "color": CH6_VARIANCE_RED, "s": CH6_W12_PUSH_MARKER_S},
            {"point": w_class, "color": "#111111", "s": CH6_W12_PUSH_MARKER_S},
        ]

    def _view_kw(b4, *, cam_u=1.0):
        return dict(
            view_w_st_lim=(float(b4[0]), float(b4[1])),
            view_w_el_lim=(float(b4[2]), float(b4[3])),
            camera_zoom=1.0 + 0.42 * float(cam_u),
        )

    def _lerp_w12(u, target4):
        u = smooth(float(u))
        tlo1, thi1, tlo2, thi2 = target4
        ax_lo, ax_hi = float(axis_lim[0]), float(axis_lim[1])
        return (
            ax_lo + u * (float(tlo1) - ax_lo),
            ax_hi + u * (float(thi1) - ax_hi),
            ax_lo + u * (float(tlo2) - ax_lo),
            ax_hi + u * (float(thi2) - ax_hi),
        )

    def _emit(**kw):
        b4 = kw.pop("w12_bounds", w12_full)
        cam_u = kw.pop("camera_zoom_u", 1.0)
        base = dict(
            xlim=info["xlim"],
            ylim=info["ylim"],
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=W,
            mu=mu,
            axis_lim=axis_lim,
            revealed_mask=np.ones(len(W), dtype=bool),
            grey_u=0.0,
            show_center=False,
            view_elev=ct_elev,
            view_azim=azim_end,
            knob_w=kw.pop("knob_w", w_knob),
            ghost_fade_u=0.0,
            marker_s=marker_s,
            belief_z_lim=belief_z,
            population_fade_u=0.0,
            matrix_fade_u=0.0,
            highlight_classroom_w=w_class,
            highlight_classroom_u=1.0,
            highlight_mu_u=1.0,
            highlight_classroom_color="#111111",
            mu_threshold_2d_color=CH6_VARIANCE_RED,
            nll_grad_3d_u=0.0,
            nll_w12_surface=class_surf,
            nll_w12_surface_alpha=CH6_W12_PUSH_SURFACE_ALPHA,
            nll_w12_alt_surface=alt_surf,
            nll_w12_alt_surface_alpha=0.30,
            nll_w12_alt_palette="belief",
            nll_w12_surface_reveal_origin=rev_origin,
            weight_grad_floor_only=True,
            weight_grad_color=CH6_VARIANCE_RED,
            surface_markers=_surface_markers(),
            surface_marker_s=CH6_W12_PUSH_MARKER_S,
            **_view_kw(b4, cam_u=cam_u),
        )
        base.update(kw)
        return _finish(_frame_population_variance_duo(**base), clip_id)

    return dict(
        emit=_emit,
        smooth=smooth,
        mu_w12=mu_w12,
        w_class=w_class,
        w_knob=w_knob,
        w12_full=w12_full,
        w12_zoom=pack["w12_zoom"],
        g_newton=pack["g_newton"],
        H_w12=pack["H_w12"],
        Xd=pack["Xd"],
        y=pack["y"],
        g_scale=g_scale,
        lerp_w12=_lerp_w12,
    )


def _build_ch6_131_population_n100_classroom_w12_push(clip_id):
    """Cloud → two points → radial NLL → zoom → gradient → low-curvature push."""
    ctx = _ch6_w12_push_emit_factory(_ch6_w12_push_scene_pack(), clip_id)
    _emit = ctx["emit"]
    smooth = ctx["smooth"]
    mu_w12 = ctx["mu_w12"]
    w_class = ctx["w_class"]
    w12_full = ctx["w12_full"]
    w12_zoom = ctx["w12_zoom"]
    g_newton = ctx["g_newton"]
    H_w12 = ctx["H_w12"]
    Xd = ctx["Xd"]
    y = ctx["y"]
    g_scale = ctx["g_scale"]
    _lerp_w12 = ctx["lerp_w12"]
    n_hold = CH6_N_HOLD * 2
    frames: list = []

    frames.extend(_hold(_emit(
        population_fade_u=1.0,
        nll_w12_surface_u=0.0,
        nll_w12_markers_only_u=0.0,
        w12_bounds=w12_full,
        camera_zoom_u=0.0,
    ), max(1, n_hold // 2)))

    n_collapse = _draft_short(28, 12)
    for k in range(n_collapse):
        u = smooth(float(k) / max(n_collapse - 1, 1))
        frames.append(_emit(
            population_fade_u=1.0 - u,
            nll_w12_surface_u=0.0,
            nll_w12_markers_only_u=u,
            w12_bounds=w12_full,
            camera_zoom_u=0.0,
        ))
    frames.extend(_hold(_emit(
        nll_w12_markers_only_u=1.0,
        w12_bounds=w12_full,
        camera_zoom_u=0.0,
    ), max(1, n_hold // 3)))

    n_reveal = _draft_short(40, 16)
    for k in range(n_reveal):
        u = smooth(float(k) / max(n_reveal - 1, 1))
        frames.append(_emit(
            nll_w12_markers_only_u=1.0,
            nll_w12_surface_u=u,
            nll_w12_surface_reveal_u=u,
            w12_bounds=w12_full,
            camera_zoom_u=0.0,
        ))
    frames.extend(_hold(frames[-1], max(1, n_hold // 3)))

    n_zoom = _draft_short(40, 16)
    for k in range(n_zoom):
        u = smooth(float(k) / max(n_zoom - 1, 1))
        frames.append(_emit(
            nll_w12_markers_only_u=1.0,
            nll_w12_surface_u=1.0,
            nll_w12_surface_reveal_u=1.0,
            w12_bounds=_lerp_w12(u, w12_zoom),
            camera_zoom_u=u,
        ))
    frames.extend(_hold(frames[-1], max(1, n_hold // 3)))

    n_grad = _draft_short(28, 10)
    for k in range(n_grad):
        u = smooth(float(k) / max(n_grad - 1, 1))
        frames.append(_emit(
            nll_w12_markers_only_u=1.0,
            nll_w12_surface_u=1.0,
            nll_w12_surface_reveal_u=1.0,
            weight_grad=g_newton,
            weight_grad_w=mu_w12,
            weight_grad_scale=g_scale * u,
            w12_bounds=w12_zoom,
            camera_zoom_u=1.0,
        ))
    frames.extend(_hold(frames[-1], n_hold))

    push_path = _ch6_curvature_push_path(
        mu_w12[:2], g_newton, H_w12,
        n_frames=_draft_short(34, 14),
        curv_mult=CH6_W12_PUSH_LOW_CURV_MULT,
    )
    for i, xy in enumerate(push_path[1:], start=1):
        w_walk = np.array([float(xy[0]), float(xy[1]), 0.0], dtype=np.float64)
        g_now = _ch6_newton_dir_w12(w_walk, Xd, y)
        arrow_u = _ch6_w12_push_arrow_scale(
            mu_w12, xy, H_w12, curv_mult=CH6_W12_PUSH_LOW_CURV_MULT,
        )
        frames.append(_emit(
            nll_w12_markers_only_u=1.0,
            nll_w12_surface_u=1.0,
            nll_w12_surface_reveal_u=1.0,
            weight_grad=g_now,
            weight_grad_w=w_walk,
            weight_grad_scale=g_scale * max(0.10, arrow_u),
            w12_bounds=w12_zoom,
            camera_zoom_u=1.0,
            mu_ghost_u=1.0,
            walker_w=w_walk,
            walker_u=1.0,
            highlight_mu_u=0.0,
        ))
    frames.extend(_hold(frames[-1], n_hold))
    return frames


def _build_ch6_132_population_n100_classroom_w12_curvature_contrast(clip_id):
    """Bend classroom NLL → rotated steep bowl; markers fixed in xy, z rides the morph."""
    pack = _ch6_w12_push_scene_pack()
    mu_w12 = pack["mu_w12"]
    w_class = pack["w_class"]
    class_zoom_geom = pack["class_zoom_geom"]
    class_zoom_vis = pack["class_zoom_vis"]
    steep_geom = pack["steep_geom"]
    steep_surf = pack["steep_surf"]
    steep_H = pack["steep_H_w12"]
    g_newton = pack["g_newton"]
    ctx = _ch6_w12_push_emit_factory(pack, clip_id)
    _emit = ctx["emit"]
    smooth = ctx["smooth"]
    w12_zoom = ctx["w12_zoom"]
    g_scale = ctx["g_scale"]
    n_hold = CH6_N_HOLD * 2

    def _markers():
        return [
            {"point": mu_w12, "color": CH6_VARIANCE_RED, "s": CH6_W12_PUSH_MARKER_S},
            {"point": w_class, "color": "#111111", "s": CH6_W12_PUSH_MARKER_S},
        ]

    def _morph_vis(u):
        geom = _ch6_w12_morph_geom(class_zoom_geom, steep_geom, u)
        return _ch6_geom_to_belief_vis(geom)

    def _base_kw(**extra):
        return dict(
            nll_w12_markers_only_u=1.0,
            nll_w12_alt_surface=None,
            nll_w12_alt_surface_u=0.0,
            w12_bounds=w12_zoom,
            camera_zoom_u=1.0,
            surface_markers=_markers(),
            mu=mu_w12,
            highlight_classroom_w=w_class,
            **extra,
        )

    def _final_kw(**extra):
        return _base_kw(
            nll_w12_surface=steep_surf,
            nll_w12_surface_u=1.0,
            nll_w12_surface_reveal_u=1.0,
            nll_w12_marker_surface=steep_surf,
            **extra,
        )

    frames: list = []

    frames.extend(_hold(_emit(**_base_kw(
        nll_w12_surface=class_zoom_vis,
        nll_w12_surface_u=1.0,
        nll_w12_surface_reveal_u=1.0,
        nll_w12_marker_surface=class_zoom_vis,
    )), max(1, n_hold // 2)))

    n_morph = _draft_short(48, 20)
    for k in range(n_morph):
        u = smooth(float(k) / max(n_morph - 1, 1))
        mv = _morph_vis(u)
        frames.append(_emit(**_base_kw(
            nll_w12_surface=mv,
            nll_w12_surface_u=1.0,
            nll_w12_surface_reveal_u=1.0,
            nll_w12_marker_surface=mv,
        )))
    frames.extend(_hold(_emit(**_final_kw()), max(1, n_hold // 3)))

    n_grad = _draft_short(24, 10)
    for k in range(n_grad):
        u = smooth(float(k) / max(n_grad - 1, 1))
        frames.append(_emit(**_final_kw(
            weight_grad=g_newton,
            weight_grad_w=mu_w12,
            weight_grad_scale=g_scale * u,
        )))
    frames.extend(_hold(frames[-1], max(1, n_hold // 3)))

    push_path = _ch6_curvature_push_path(
        mu_w12[:2], g_newton, steep_H,
        n_frames=_draft_short(38, 16),
        curv_mult=CH6_W12_PUSH_HIGH_CURV_MULT * 1.5,
        step_eta=CH6_W12_PUSH_STEP_ETA * 1.45,
    )
    for xy in push_path[1:]:
        w_walk = np.array([float(xy[0]), float(xy[1]), 0.0], dtype=np.float64)
        arrow_u = _ch6_w12_push_arrow_scale(
            mu_w12, xy, steep_H, curv_mult=CH6_W12_PUSH_HIGH_CURV_MULT * 1.5,
        )
        frames.append(_emit(**_final_kw(
            weight_grad=g_newton,
            weight_grad_w=w_walk,
            weight_grad_scale=g_scale * max(0.08, arrow_u),
            mu_ghost_u=1.0,
            walker_w=w_walk,
            walker_u=1.0,
            highlight_mu_u=0.0,
        )))
    frames.extend(_hold(frames[-1], n_hold * 2))
    return frames


def _build_ch6_129_population_n100_classroom_spring_from_127(clip_id, *, voxel_trail=False):
    """Core builder for ch6_129 / ch6_130."""
    pose = _ch6_matrix_contrast_cloud()
    info = pose["info"]
    mu = np.asarray(pose["mu"], dtype=np.float64).reshape(3)
    W = np.asarray(pose["W"], dtype=np.float64)
    axis_lim = pose["axis_lim"]
    marker_s = float(info.get("marker_s", 28))
    belief_z = info.get("belief_z_lim")
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2

    study = np.asarray(info["cs"], dtype=np.float64)
    exam = np.asarray(info["ce"], dtype=np.float64)
    y = np.asarray(info["cy"], dtype=np.float64).reshape(-1)
    panel_xlim, panel_ylim = info["xlim"], info["ylim"]
    w_class = np.asarray(info["w_knob"], dtype=np.float64).reshape(3)
    w_target = w_class.copy()

    ax_lo, ax_hi = float(axis_lim[0]), float(axis_lim[1])
    full_bounds = (ax_lo, ax_hi, ax_lo, ax_hi, ax_lo, ax_hi)
    b_zoom_lo, b_zoom_hi = (float(v) for v in CH6_SPRING_ZOOM_BIAS_BOUNDS)
    zoom_bounds = _ch6_voxel_bounds_w12_between_points(mu, w_class, axis_lim, pad_frac=0.15)
    zoom_bounds = (
        zoom_bounds[0], zoom_bounds[1], zoom_bounds[2], zoom_bounds[3],
        b_zoom_lo, b_zoom_hi,
    )
    w12_span = float(zoom_bounds[1]) - float(zoom_bounds[0])
    b_span = float(b_zoom_hi) - float(b_zoom_lo)
    voxel_cache = _ch6_nll_voxel_cache(study, exam, y, bounds=full_bounds)
    nll_vmin, nll_vmax = _g("ch4_nll_global_scale")()

    ct_elev = ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV")))
    azim_end = float(_g("CH3_LIK_W12_CT_AZIM")) + 720.0
    ref_max = float(CH6_NLL_VOXEL_CURV_REF_PCT_MAX)

    Xd = ch6_design(study, exam)
    H = ch6_observed_information(mu, Xd, y, ridge=CH6_RIDGE)
    g_mu = ch6_nll_grad(mu, Xd, y, ridge=CH6_RIDGE)
    newton_dir_mu = _ch6_newton_ascent_dir(mu, Xd, y)
    grad_span = 0.052
    grad_3d_scale = 0.20 * w12_span
    grad_3d_axis_lim = (zoom_bounds[0], zoom_bounds[1])

    dim_mask = np.zeros(len(W), dtype=bool)

    def _lerp_bounds(u, target_bounds):
        u = float(u)
        tlo1, thi1, tlo2, thi2, tlob, thib = target_bounds
        return (
            float(ax_lo) + u * (float(tlo1) - float(ax_lo)),
            float(ax_hi) + u * (float(thi1) - float(ax_hi)),
            float(ax_lo) + u * (float(tlo2) - float(ax_lo)),
            float(ax_hi) + u * (float(thi2) - float(ax_hi)),
            float(ax_lo) + u * (float(tlob) - float(ax_lo)),
            float(ax_hi) + u * (float(thib) - float(ax_hi)),
        )

    def _pair_kw(*, dim_u=1.0, mu_u=1.0, class_u=1.0):
        du = float(dim_u)
        return dict(
            marker_bright_mask=dim_mask if du > 1e-4 else None,
            marker_dim_scale=0.16 if du > 1e-4 else 1.0,
            highlight_mu_u=float(mu_u),
            highlight_classroom_u=float(class_u),
            highlight_classroom_color="#111111",
            show_center=True,
        )

    def _callouts(avg_u=0.0, class_u=0.0):
        out = []
        au, cu = float(avg_u), float(class_u)
        if au > 1e-4:
            out.append(dict(
                point=mu, label="average",
                color=CH6_VARIANCE_RED, u=au,
                label_placement="bottom_outside",
            ))
        if cu > 1e-4:
            out.append(dict(
                point=w_class, label="this classroom",
                color=CH6_VARIANCE_RED, u=cu,
                label_placement="top_outside",
            ))
        return out or None

    def _emit(**kw):
        base = dict(
            xlim=panel_xlim,
            ylim=panel_ylim,
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=W,
            mu=mu,
            axis_lim=axis_lim,
            revealed_mask=np.ones(len(W), dtype=bool),
            grey_u=0.0,
            show_center=False,
            view_elev=ct_elev,
            view_azim=azim_end,
            knob_w=kw.pop("knob_w", w_class),
            ghost_fade_u=0.0,
            marker_s=marker_s,
            belief_z_lim=belief_z,
            population_fade_u=1.0,
            matrix_fade_u=0.0,
            nll_voxel_cache=voxel_cache,
            voxel_view_bounds=full_bounds,
            highlight_classroom_w=w_class,
            mu_threshold_2d_color=CH6_VARIANCE_RED,
            mu_threshold_2d_lw=3.6,
            ghost_threshold_2d_color=CH6_VARIANCE_RED,
            grad_span_frac=grad_span,
            nll_grad_3d=newton_dir_mu,
            nll_grad_3d_w=mu,
            nll_grad_3d_scale=grad_3d_scale,
            nll_grad_3d_axis_lim=grad_3d_axis_lim,
            nll_grad_3d_is_direction=True,
        )
        base.update(kw)
        return _finish(_frame_population_variance_duo(**base), clip_id)

    frames: list = []

    frames.extend(_hold(_emit(
        nll_voxel_sweep_u=1.0,
        nll_voxel_alpha_u=1.0,
        nll_voxel_curv_ref_pct=ref_max,
        population_fade_u=0.0,
        voxel_view_bounds=full_bounds,
    ), n_hold))

    n_vox_out = _draft_short(36, 14)
    for k in range(n_vox_out):
        u = smooth(float(k) / max(n_vox_out - 1, 1))
        frames.append(_emit(
            nll_voxel_sweep_u=1.0,
            nll_voxel_alpha_u=1.0 - u,
            nll_voxel_curv_ref_pct=ref_max,
            population_fade_u=u,
            voxel_view_bounds=full_bounds,
        ))

    frames.extend(_hold(_emit(
        population_fade_u=1.0,
        nll_voxel_sweep_u=0.0,
        voxel_view_bounds=full_bounds,
    ), n_hold))

    n_grey = _draft_short(28, 10)
    for k in range(n_grey):
        u = smooth(float(k) / max(n_grey - 1, 1))
        frames.append(_emit(
            population_fade_u=1.0,
            highlight_classroom_w=w_class,
            voxel_view_bounds=full_bounds,
            **_pair_kw(dim_u=u, mu_u=u, class_u=u),
        ))
    frames.extend(_hold(frames[-1], max(1, n_hold // 2)))

    n_zoom = _draft_short(40, 16)
    for k in range(n_zoom):
        u = smooth(float(k) / max(n_zoom - 1, 1))
        frames.append(_emit(
            population_fade_u=1.0,
            highlight_classroom_w=w_class,
            voxel_view_bounds=_lerp_bounds(u, zoom_bounds),
            **_pair_kw(),
        ))
    frames.extend(_hold(frames[-1], max(1, n_hold // 3)))

    n_lbl_avg = _draft_short(24, 10)
    for k in range(n_lbl_avg):
        u = smooth(float(k) / max(n_lbl_avg - 1, 1))
        frames.append(_emit(
            population_fade_u=1.0,
            highlight_classroom_w=w_class,
            voxel_view_bounds=zoom_bounds,
            ch6_callouts=_callouts(avg_u=u),
            **_pair_kw(),
        ))
    frames.extend(_hold(frames[-1], max(1, n_hold // 4)))

    n_lbl_class = _draft_short(24, 10)
    for k in range(n_lbl_class):
        u = smooth(float(k) / max(n_lbl_class - 1, 1))
        frames.append(_emit(
            population_fade_u=1.0,
            highlight_classroom_w=w_class,
            voxel_view_bounds=zoom_bounds,
            ch6_callouts=_callouts(avg_u=1.0, class_u=u),
            **_pair_kw(),
        ))
    frames.extend(_hold(frames[-1], max(1, n_hold // 4)))

    n_dist = _draft_short(32, 12)
    for k in range(n_dist):
        u = smooth(float(k) / max(n_dist - 1, 1))
        frames.append(_emit(
            population_fade_u=1.0,
            highlight_classroom_w=w_class,
            ch6_pair_line_u=u,
            ch6_pair_line_alpha=1.0,
            voxel_view_bounds=zoom_bounds,
            ch6_callouts=_callouts(avg_u=1.0, class_u=1.0),
            **_pair_kw(),
        ))
    frames.extend(_hold(frames[-1], max(1, n_hold // 3)))

    n_grad3d = _draft_short(28, 10)
    for k in range(n_grad3d):
        u = smooth(float(k) / max(n_grad3d - 1, 1))
        frames.append(_emit(
            population_fade_u=1.0,
            highlight_classroom_w=w_class,
            ch6_pair_line_u=1.0,
            ch6_pair_line_alpha=1.0,
            nll_grad_3d_u=u,
            voxel_view_bounds=zoom_bounds,
            ch6_callouts=_callouts(avg_u=1.0, class_u=1.0),
            **_pair_kw(),
        ))
    frames.extend(_hold(frames[-1], max(1, n_hold // 3)))

    n_dist_out = _draft_short(20, 8)
    for k in range(n_dist_out):
        u = smooth(float(k) / max(n_dist_out - 1, 1))
        frames.append(_emit(
            population_fade_u=1.0,
            highlight_classroom_w=w_class,
            ch6_pair_line_u=1.0,
            ch6_pair_line_alpha=1.0 - u,
            mu_threshold_2d=mu,
            mu_threshold_2d_u=u,
            knob_w=mu,
            show_2d_grads=u > 0.35,
            grad_w_live=mu,
            nll_grad_3d_u=1.0,
            voxel_view_bounds=zoom_bounds,
            ch6_callouts=_callouts(avg_u=1.0, class_u=1.0),
            **_pair_kw(),
        ))
    frames.extend(_hold(_emit(
        population_fade_u=1.0,
        highlight_classroom_w=w_class,
        mu_threshold_2d=mu,
        mu_threshold_2d_u=1.0,
        knob_w=mu,
        show_2d_grads=True,
        grad_w_live=mu,
        nll_grad_3d_u=1.0,
        voxel_view_bounds=zoom_bounds,
        ch6_callouts=_callouts(avg_u=1.0, class_u=1.0),
        **_pair_kw(),
    ), n_hold))

    spring_path = _hessian_restore_path(
        mu[:2], w_target[:2], H[:2, :2],
        n_frames=_draft_short(28, 12),
        damp=1.15,
        stiffness=5.0,
    )
    newton_path_3d = _ch6_damped_newton_path_3d(
        mu, w_target, Xd, y,
        n_frames=len(spring_path),
        axis_lim=axis_lim,
        pull_gain=0.55,
    )
    newton_path_3d[-1] = w_target.copy()
    trail_cell_w12, trail_cell_b, trail_hc_w12, trail_hc_b = (
        _ch6_nll_voxel_trail_params(w12_span, b_span) if voxel_trail else (0.0, 0.0, 0, 0)
    )
    voxel_trail_draws: list = []
    trail_keys: set = set()

    def _trail_merge_draw():
        if not voxel_trail_draws:
            return None
        return _ch6_ball_voxel_merge_draws(voxel_trail_draws)

    def _trail_append(w_pos):
        if not voxel_trail:
            return
        draw = _ch6_nll_voxel_trail_cluster(
            w_pos, study, exam, y,
            cell_w12=trail_cell_w12,
            cell_b=trail_cell_b,
            half_cells_w12=trail_hc_w12,
            half_cells_b=trail_hc_b,
            vmin=nll_vmin,
            vmax=nll_vmax,
        )
        if draw is None:
            return
        key = draw["cell_key"]
        if key in trail_keys:
            return
        trail_keys.add(key)
        voxel_trail_draws.append(draw)

    for i in range(1, len(spring_path)):
        t = float(i) / max(len(spring_path) - 1, 1)
        xy = spring_path[i]
        w_walk = np.asarray(newton_path_3d[i], dtype=np.float64).reshape(3).copy()
        if i == len(spring_path) - 1:
            w_walk = w_target.copy()
        ww_line = mu.copy()
        ww_line[0], ww_line[1] = float(xy[0]), float(xy[1])
        ww_line[2] = (1.0 - t) * float(mu[2]) + t * float(w_target[2])
        if i == len(spring_path) - 1:
            ww_line = w_target.copy()
        newton_dir = _ch6_newton_ascent_dir(w_walk, Xd, y)
        _trail_append(w_walk)
        spring_kw = dict(
            population_fade_u=1.0,
            highlight_classroom_w=w_class,
            mu_threshold_2d=mu,
            mu_threshold_2d_u=1.0,
            ghost_threshold_2d=mu,
            ghost_threshold_2d_u=1.0,
            classroom_threshold_2d=ww_line,
            classroom_threshold_2d_u=1.0,
            knob_w=ww_line,
            show_2d_grads=True,
            grad_w_live=ww_line,
            nll_grad_3d=newton_dir,
            nll_grad_3d_w=w_walk,
            nll_grad_3d_u=max(0.12, 1.0 - t * 0.82),
            nll_grad_3d_is_direction=True,
            voxel_view_bounds=zoom_bounds,
            ch6_callouts=_callouts(avg_u=1.0, class_u=1.0),
            **_pair_kw(mu_u=0.0, class_u=0.0),
            mu_ghost_u=1.0,
            walker_w=w_walk,
            walker_u=1.0,
        )
        if voxel_trail:
            spring_kw["nll_voxel_trail"] = _trail_merge_draw()
        frames.append(_emit(**spring_kw))

    end_kw = dict(
        population_fade_u=1.0,
        highlight_classroom_w=w_class,
        mu_threshold_2d=mu,
        mu_threshold_2d_u=1.0,
        ghost_threshold_2d=mu,
        ghost_threshold_2d_u=1.0,
        classroom_threshold_2d=w_target,
        classroom_threshold_2d_u=1.0,
        knob_w=w_target,
        show_2d_grads=True,
        grad_w_live=w_target,
        nll_grad_3d_u=0.0,
        voxel_view_bounds=zoom_bounds,
        ch6_callouts=_callouts(avg_u=1.0, class_u=1.0),
        **_pair_kw(mu_u=0.0, class_u=1.0),
        mu_ghost_u=1.0,
        walker_u=0.0,
    )
    if voxel_trail:
        _trail_append(w_target)
        end_kw["nll_voxel_trail"] = _trail_merge_draw()
    frames.extend(_hold(_emit(**end_kw), n_hold))
    return frames


def _ch6_voxel_medium_demo_scene(
    w_start=None,
    w_target=None,
    *,
    axis_lim=None,
    pad_frac=0.18,
):
    """Custom bounds + pair layout for ch6_134+ (independent of classroom zoom)."""
    if w_start is None:
        w_start = CH6_VOXEL_MEDIUM_DEMO_W_START
    if w_target is None:
        w_target = CH6_VOXEL_MEDIUM_DEMO_W_TARGET
    if axis_lim is None:
        axis_lim = CH6_VOXEL_MEDIUM_DEMO_AXIS_LIM
    w0 = np.asarray(w_start, dtype=np.float64).reshape(3)
    w1 = np.asarray(w_target, dtype=np.float64).reshape(3)
    bounds = _ch6_voxel_bounds_cube_between_points(w0, w1, axis_lim, pad_frac=pad_frac)
    bowl_mu = 0.5 * (w0 + w1)
    bowl_mu[2] = 0.0
    return dict(
        w_start=w0,
        w_target=w1,
        bounds=bounds,
        axis_lim=axis_lim,
        bowl_mu=bowl_mu,
        w12_span=float(bounds[1]) - float(bounds[0]),
    )


def _ch6_voxel_medium_push_pack():
    """Shared pose, bounds, and Hessian for voxel-medium push demos."""
    pose = _ch6_matrix_contrast_cloud()
    info = pose["info"]
    mu = np.asarray(pose["mu"], dtype=np.float64).reshape(3)
    w_class = np.asarray(info["w_knob"], dtype=np.float64).reshape(3)
    axis_lim = pose["axis_lim"]
    b_zoom_lo, b_zoom_hi = (float(v) for v in CH6_SPRING_ZOOM_BIAS_BOUNDS)
    zoom_bounds = _ch6_voxel_bounds_w12_between_points(mu, w_class, axis_lim, pad_frac=0.15)
    zoom_bounds = (
        zoom_bounds[0], zoom_bounds[1], zoom_bounds[2], zoom_bounds[3],
        b_zoom_lo, b_zoom_hi,
    )
    study = np.asarray(info["cs"], dtype=np.float64)
    exam = np.asarray(info["ce"], dtype=np.float64)
    y = np.asarray(info["cy"], dtype=np.float64).reshape(-1)
    Xd = ch6_design(study, exam)
    H = ch6_observed_information(mu, Xd, y, ridge=CH6_RIDGE)
    return dict(
        pose=pose,
        info=info,
        mu=mu,
        w_class=w_class,
        axis_lim=axis_lim,
        zoom_bounds=zoom_bounds,
        study=study,
        exam=exam,
        y=y,
        Xd=Xd,
        H=H,
        w12_span=float(zoom_bounds[1]) - float(zoom_bounds[0]),
    )


def _ch6_push_path_3d(w0, w1, xy_path):
    """Lift a 2-D push path into 3-D with linear bias interpolation."""
    w0 = np.asarray(w0, dtype=np.float64).reshape(3)
    w1 = np.asarray(w1, dtype=np.float64).reshape(3)
    out = []
    n = len(xy_path)
    for i, xy in enumerate(xy_path):
        t = float(i) / max(n - 1, 1)
        w = w0.copy()
        w[0], w[1] = float(xy[0]), float(xy[1])
        w[2] = (1.0 - t) * float(w0[2]) + t * float(w1[2])
        out.append(w)
    return out


def _ch6_grad_current_scale(w0, w, H, *, curv_mult=1.0):
    """How strongly the local gradient current pushes (decays with curvature)."""
    d = np.asarray(w[:2], dtype=np.float64) - np.asarray(w0[:2], dtype=np.float64)
    H_eff = float(curv_mult) * np.asarray(H[:2, :2], dtype=np.float64)
    evals = np.linalg.eigvalsh(H_eff)
    floor = max(0.08, -float(np.min(evals)) + 0.08)
    H_eff = H_eff + floor * np.eye(2, dtype=np.float64)
    return float(np.exp(-0.62 * float(d @ H_eff @ d)))


def _ch6_voxel_medium_chord_path(w0, w_dest, bounds, *, n_frames):
    """Straight in-bounds chord (flat medium — no wall-sliding)."""
    w0 = _ch6_clip_w_to_bounds(w0, bounds)
    w_dest = _ch6_clip_w_to_bounds(w_dest, bounds)
    n = max(int(n_frames), 2)
    out = []
    for i in range(n):
        t = float(i) / float(n - 1)
        out.append(_ch6_clip_w_to_bounds(w0 + t * (w_dest - w0), bounds))
    out[-1] = w_dest.copy()
    return out


def _ch6_voxel_medium_comparable_push(
    w0,
    w_goal,
    H_push,
    *,
    snap_target=False,
    step_eta=None,
    max_steps=None,
    stall_tol=2.5e-4,
    clip_bounds=None,
):
    """Identical integrator for 134+; only ``H_push`` changes how fast the push decays."""
    w0 = np.asarray(w0, dtype=np.float64).reshape(3)
    w_goal = np.asarray(w_goal, dtype=np.float64).reshape(3)
    H = np.asarray(H_push, dtype=np.float64).reshape(3, 3)
    n_push = int(max_steps) if max_steps is not None else _draft_short(96, 38)
    eta = float(CH6_VOXEL_MEDIUM_PUSH_ETA if step_eta is None else step_eta)
    g_push = w_goal[:2] - w0[:2]
    gn = float(np.linalg.norm(g_push))
    if clip_bounds is not None:
        dlo1, dhi1, dlo2, dhi2 = (float(clip_bounds[i]) for i in range(4))
    else:
        dlo1, dhi1, dlo2, dhi2 = CH6_VIEW_BOUNDS_W12
    if gn < 1e-12:
        xy_path = np.tile(w0[:2].reshape(1, 2), (2, 1))
    else:
        u_dir = g_push / gn
        H_eff = H[:2, :2].copy()
        evals = np.linalg.eigvalsh(H_eff)
        floor = max(0.08, -float(np.min(evals)) + 0.08)
        H_eff = H_eff + floor * np.eye(2, dtype=np.float64)
        p0 = w0[:2].copy()
        path = [p0.copy()]
        p = p0.copy()
        stall_count = 0
        for _ in range(max(n_push - 1, 1)):
            along = float(np.dot(p - p0, u_dir))
            if along >= gn - 1e-5:
                p_goal = p0 + gn * u_dir
                p_goal[0] = float(np.clip(p_goal[0], dlo1, dhi1))
                p_goal[1] = float(np.clip(p_goal[1], dlo2, dhi2))
                path.append(p_goal.copy())
                break
            d = p - p0
            decay = float(np.exp(-0.62 * float(d @ H_eff @ d)))
            step = eta * decay
            p_new = p + step * u_dir
            p_new[0] = float(np.clip(p_new[0], dlo1, dhi1))
            p_new[1] = float(np.clip(p_new[1], dlo2, dhi2))
            along_new = float(np.dot(p_new - p0, u_dir))
            disp = float(np.linalg.norm(p_new - p))
            if along_new <= along + 1e-9 and disp < max(float(stall_tol), 5e-4):
                stall_count += 1
                if stall_count >= 2:
                    break
            elif disp < float(stall_tol) or step < 1e-5:
                stall_count += 1
                if stall_count >= 3:
                    break
            else:
                stall_count = 0
            p = p_new
            path.append(p.copy())
        xy_path = np.asarray(path, dtype=np.float64)
    path_3d = _ch6_push_path_3d(w0, w_goal, xy_path)
    if clip_bounds is not None:
        path_3d = _ch6_clip_path_to_bounds(path_3d, clip_bounds)
    if snap_target:
        path_3d[-1] = _ch6_clip_w_to_bounds(w_goal, clip_bounds) if clip_bounds is not None else w_goal.copy()
    return path_3d, H


def _ch6_voxel_medium_classroom_scene():
    """Shared start (average μ) and goal (classroom) for ch6_133 / 134+."""
    pack = _ch6_voxel_medium_push_pack()
    w0 = np.asarray(pack["mu"], dtype=np.float64).reshape(3)
    w_goal = np.asarray(pack["w_class"], dtype=np.float64).reshape(3)
    return dict(
        pack=pack,
        w_start=w0,
        w_goal=w_goal,
        w_target=w_goal,
        bounds=pack["zoom_bounds"],
        w12_span=pack["w12_span"],
    )


def _ch6_voxel_medium_thin_path(path_3d, *, min_step=1.2e-4):
    """Drop duplicate stalls so short pushes don't pad out to the same frame count."""
    if not path_3d:
        return path_3d
    out = [np.asarray(path_3d[0], dtype=np.float64).reshape(3)]
    for w in path_3d[1:]:
        w = np.asarray(w, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(w[:2] - out[-1][:2])) >= float(min_step):
            out.append(w)
    if len(out) < 2:
        out.append(np.asarray(path_3d[-1], dtype=np.float64).reshape(3))
    return out


def _ch6_voxel_medium_trim_stalled_tail(path_3d, *, tail_frac=0.05):
    """End the push once per-step motion falls below a fraction of the opening step."""
    if len(path_3d) < 3:
        return path_3d
    pts = [np.asarray(w, dtype=np.float64).reshape(3) for w in path_3d]
    steps = [
        float(np.linalg.norm(pts[i][:2] - pts[i - 1][:2]))
        for i in range(1, len(pts))
    ]
    if not steps:
        return pts
    thresh = max(float(tail_frac) * float(steps[0]), 1.2e-4)
    cut = len(pts)
    stall = 0
    for i, step in enumerate(steps, start=1):
        if step < thresh:
            stall += 1
            if stall >= 2:
                cut = i
                break
        else:
            stall = 0
    return pts[:cut]


def _build_ch6_voxel_medium_curvature_scenario(
    clip_id,
    *,
    kind,
    snap_target,
    end_hold_mult=2.0,
    flip_endpoints=False,
    movement_orbit_deg=0.0,
):
    """One shared classroom pose; curvature alone sets how far the red point travels."""
    scene = _ch6_voxel_medium_classroom_scene()
    w0 = scene["w_start"]
    w_class = scene["w_goal"]
    bounds = scene["bounds"]
    w_push = _ch6_chord_goal_in_bounds(w0, w_class, bounds)
    H_push = _ch6_landscape_push_hessian(w0, w_push, kind)
    if kind == "flat_plateau":
        n_flat = _draft_short(28, 12)
        path_3d = _ch6_voxel_medium_chord_path(w0, w_push, bounds, n_frames=n_flat)
        H_eff = H_push
    else:
        path_3d, H_eff = _ch6_voxel_medium_comparable_push(
            w0, w_push, H_push,
            snap_target=snap_target,
            clip_bounds=bounds,
        )
        path_3d = _ch6_voxel_medium_trim_stalled_tail(path_3d)
        path_3d = _ch6_voxel_medium_thin_path(path_3d)
    path_3d = _ch6_clip_path_to_bounds(path_3d, bounds)
    w_dest = _ch6_clip_w_to_bounds(path_3d[-1], bounds)
    if kind == "immobile_valley":
        w_dest = _ch6_immobile_near_dest(w0, w_push, bounds)
        if len(path_3d) >= 2:
            path_3d[-1] = w_dest.copy()
        else:
            path_3d = [w0.copy(), w_dest.copy()]
    else:
        path_3d[-1] = w_dest.copy()
    w_min = w0 if kind == "immobile_valley" else w_dest
    cache = _ch6_landscape_voxel_cache(
        bounds, w0, w_min, kind, H_push=H_push,
        w_dest=w_dest if kind == "immobile_valley" else None,
    )
    w_start_vis = np.asarray(w0, dtype=np.float64).reshape(3)
    w_dest_vis = np.asarray(w_dest, dtype=np.float64).reshape(3)
    path_vis = [np.asarray(w, dtype=np.float64).reshape(3) for w in path_3d]
    grad_fn = lambda w: w_push - np.asarray(w, dtype=np.float64).reshape(3)
    grad_anchor = w0
    if flip_endpoints:
        w_start_vis = np.asarray(w_dest, dtype=np.float64).reshape(3).copy()
        w_dest_vis = np.asarray(w0, dtype=np.float64).reshape(3).copy()
        path_vis = list(reversed(path_vis))
        path_vis[0] = w_start_vis.copy()
        path_vis[-1] = w_dest_vis.copy()
        grad_fn = lambda w: np.asarray(w, dtype=np.float64).reshape(3) - w_push
        grad_anchor = w_start_vis
    return _build_ch6_voxel_medium_push_demo(
        clip_id,
        voxel_cache=cache,
        bounds=bounds,
        w_start=w_start_vis,
        w_dest=w_dest_vis,
        w_push=w_push,
        w12_span=scene["w12_span"],
        path_3d=path_vis,
        grad_fn=grad_fn,
        grad_scale_fn=lambda w, t: _ch6_grad_current_scale(
            grad_anchor, w, H_eff, curv_mult=1.0,
        ),
        end_hold_mult=end_hold_mult,
        movement_orbit_deg=movement_orbit_deg,
    )


def _build_ch6_voxel_medium_push_demo(
    clip_id,
    *,
    voxel_cache,
    bounds,
    w_start,
    w_dest,
    w_push,
    path_3d=None,
    xy_path=None,
    grad_fn,
    grad_scale_fn,
    end_hold_mult=1.0,
    w12_span=None,
    movement_orbit_deg=0.0,
):
    """Voxel medium + gradient current + push with opaque wake near the walker."""
    pack = _ch6_voxel_medium_push_pack()
    info = pack["info"]
    axis_lim = pack["axis_lim"]
    belief_z = info.get("belief_z_lim")
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    panel_xlim, panel_ylim = info["xlim"], info["ylim"]
    marker_s = float(info.get("marker_s", 28))
    ct_elev = ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV")))
    azim_end = float(_g("CH3_LIK_W12_CT_AZIM")) + 720.0
    grad_span = 0.052
    span_w12 = float(w12_span) if w12_span is not None else float(bounds[1]) - float(bounds[0])
    grad_3d_scale = 0.20 * span_w12
    grad_3d_axis_lim = (bounds[0], bounds[1])
    w_start = np.asarray(w_start, dtype=np.float64).reshape(3)
    w_dest = _ch6_clip_w_to_bounds(w_dest, bounds)
    w_push = np.asarray(w_push, dtype=np.float64).reshape(3)
    if path_3d is None:
        if xy_path is None:
            raise ValueError("path_3d or xy_path required")
        path_3d = _ch6_push_path_3d(w_start, w_dest, xy_path)
    path_3d = [np.asarray(w, dtype=np.float64).reshape(3) for w in path_3d]
    path_3d = _ch6_clip_path_to_bounds(path_3d, bounds)
    path_3d[-1] = w_dest.copy()
    grad0 = np.asarray(grad_fn(w_start), dtype=np.float64).reshape(3)

    def _emit(**kw):
        base = dict(
            xlim=panel_xlim,
            ylim=panel_ylim,
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=pack["pose"]["W"],
            mu=w_start,
            axis_lim=axis_lim,
            revealed_mask=np.ones(len(pack["pose"]["W"]), dtype=bool),
            grey_u=0.0,
            show_center=False,
            view_elev=ct_elev,
            view_azim=azim_end,
            knob_w=w_dest,
            ghost_fade_u=0.0,
            marker_s=marker_s,
            belief_z_lim=belief_z,
            population_fade_u=1.0,
            marker_alpha_scale=0.0,
            matrix_fade_u=0.0,
            highlight_mu_u=1.0,
            highlight_mu_w=w_start,
            highlight_classroom_w=w_dest,
            highlight_classroom_u=1.0,
            highlight_classroom_color="#111111",
            ch6_pair_line_u=1.0,
            ch6_pair_line_alpha=1.0,
            voxel_view_bounds=bounds,
            nll_voxel_cache=voxel_cache,
            nll_voxel_sweep_u=1.0,
            nll_voxel_alpha_u=1.0,
            nll_voxel_uniform_alpha=float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA),
            grad_span_frac=grad_span,
            nll_grad_3d=grad0,
            nll_grad_3d_w=w_start,
            nll_grad_3d_scale=grad_3d_scale,
            nll_grad_3d_axis_lim=grad_3d_axis_lim,
            nll_grad_3d_is_direction=True,
            nll_grad_3d_u=0.0,
        )
        base.update(kw)
        return _finish(_frame_population_variance_duo(**base), clip_id)

    frames: list = []
    frames.extend(_hold(_emit(
        highlight_mu_w=w_start,
        nll_voxel_sweep_u=0.0,
    ), max(1, n_hold // 2)))

    n_vox = _draft_short(40, 16)
    for k in range(n_vox):
        u = smooth(float(k) / max(n_vox - 1, 1))
        frames.append(_emit(
            highlight_mu_w=w_start,
            nll_voxel_sweep_u=u,
        ))
    frames.extend(_hold(frames[-1], max(1, n_hold // 3)))

    n_grad = _draft_short(24, 10)
    for k in range(n_grad):
        u = smooth(float(k) / max(n_grad - 1, 1))
        frames.append(_emit(
            highlight_mu_w=w_start,
            nll_grad_3d_u=u,
            nll_grad_3d_scale=grad_3d_scale * grad_scale_fn(w_start, 0.0),
        ))
    frames.extend(_hold(frames[-1], max(1, n_hold // 3)))

    chord = float(np.linalg.norm(w_dest[:2] - w_start[:2]))
    n_move = max(len(path_3d) - 1, 1)
    orbit_deg = float(movement_orbit_deg)
    for i in range(1, len(path_3d)):
        w_walk = _ch6_clip_w_to_bounds(path_3d[i], bounds)
        g_now = np.asarray(grad_fn(w_walk), dtype=np.float64).reshape(3)
        traveled = float(np.linalg.norm(w_walk[:2] - w_start[:2]))
        t = traveled / max(chord, 1e-9)
        g_scale = float(grad_scale_fn(w_walk, t))
        opaque_u = float(CH6_NLL_VOXEL_WALKER_OPAQUE_MAX) * smooth(
            0.12 + 0.88 * min(1.0, traveled / max(0.06 * chord, 1e-9)),
        )
        move_kw = dict(
            highlight_mu_w=w_walk,
            highlight_classroom_w=w_dest,
            highlight_classroom_u=1.0,
            ch6_pair_line_u=1.0,
            ch6_pair_line_alpha=1.0,
            nll_grad_3d=g_now,
            nll_grad_3d_w=w_walk,
            nll_grad_3d_u=max(0.08, g_scale),
            nll_grad_3d_scale=grad_3d_scale * max(0.06, g_scale),
            nll_voxel_walker_w=w_walk,
            nll_voxel_walker_opaque_u=opaque_u,
        )
        if orbit_deg > 1e-6:
            orbit_u = smooth(float(i - 1) / max(n_move - 1, 1))
            move_kw["view_azim"] = _ch6_orbit_view_azim(
                azim_end, move_u=orbit_u, orbit_deg=orbit_deg,
            )
        frames.append(_emit(**move_kw))

    n_end = int(n_hold * max(float(end_hold_mult), 0.5))
    if orbit_deg > 1e-6 and n_end > 0:
        for k in range(n_end):
            ru = smooth(float(k) / max(n_end - 1, 1))
            frames.append(_emit(
                highlight_mu_w=w_dest,
                highlight_classroom_w=w_dest,
                highlight_classroom_u=1.0,
                ch6_pair_line_u=1.0,
                ch6_pair_line_alpha=1.0,
                nll_grad_3d_u=0.0,
                view_azim=_ch6_orbit_view_azim(
                    azim_end, return_u=ru, orbit_deg=orbit_deg,
                ),
            ))
    else:
        frames.extend(_hold(_emit(
            highlight_mu_w=w_dest,
            highlight_classroom_w=w_dest,
            highlight_classroom_u=1.0,
            ch6_pair_line_u=1.0,
            ch6_pair_line_alpha=1.0,
            nll_grad_3d_u=0.0,
        ), n_end))
    return frames


def _build_ch6_134_voxel_medium_flat_current(clip_id):
    """Low curvature — push decays slowly, point travels far to the classroom."""
    return _build_ch6_voxel_medium_curvature_scenario(
        clip_id,
        kind="flat_plateau",
        snap_target=True,
        end_hold_mult=1.6,
    )


def _build_ch6_135_voxel_medium_steep_current(clip_id):
    """High curvature — push dies quickly, point stops partway along the chord."""
    return _build_ch6_voxel_medium_curvature_scenario(
        clip_id,
        kind="steep_decay",
        snap_target=False,
        end_hold_mult=3.5,
        flip_endpoints=True,
    )


def _build_ch6_136_voxel_medium_shrinking_current_classroom(clip_id):
    """Tight valley — even a strong initial push decays before any real travel."""
    return _build_ch6_voxel_medium_curvature_scenario(
        clip_id,
        kind="immobile_valley",
        snap_target=False,
        end_hold_mult=4.0,
    )


def build_ch6_134_voxel_medium_flat_current(clip_id):
    """Flat convex voxel medium — point travels far on a strong current."""
    return _build_ch6_134_voxel_medium_flat_current(clip_id)


def build_ch6_135_voxel_medium_steep_current(clip_id):
    """Steep convex voxel medium — point stalls as the current weakens."""
    return _build_ch6_135_voxel_medium_steep_current(clip_id)


def build_ch6_136_voxel_medium_shrinking_current_classroom(clip_id):
    """Classroom NLL medium — shrinking gradient current along the push."""
    return _build_ch6_136_voxel_medium_shrinking_current_classroom(clip_id)


def build_ch6_137_population_n100_classroom_spring_voxel_resistance_orbit_from_133(clip_id):
    """ch6_133 with a 90° orbit during the spring push, then unwind."""
    return _build_ch6_133_population_n100_classroom_spring_voxel_resistance_from_129(
        clip_id, movement_orbit_deg=90.0,
    )


def build_ch6_138_voxel_medium_flat_current_orbit(clip_id):
    """ch6_134 with a 90° orbit during the push, then unwind."""
    return _build_ch6_voxel_medium_curvature_scenario(
        clip_id,
        kind="flat_plateau",
        snap_target=True,
        end_hold_mult=1.6,
        movement_orbit_deg=90.0,
    )


def build_ch6_139_voxel_medium_steep_current_orbit(clip_id):
    """ch6_135 with a 90° orbit during the push, then unwind."""
    return _build_ch6_voxel_medium_curvature_scenario(
        clip_id,
        kind="steep_decay",
        snap_target=False,
        end_hold_mult=3.5,
        flip_endpoints=True,
        movement_orbit_deg=90.0,
    )


def build_ch6_140_voxel_medium_shrinking_current_classroom_orbit(clip_id):
    """ch6_136 with a 90° orbit during the push, then unwind."""
    return _build_ch6_voxel_medium_curvature_scenario(
        clip_id,
        kind="immobile_valley",
        snap_target=False,
        end_hold_mult=4.0,
        movement_orbit_deg=90.0,
    )


def _build_ch6_133_population_n100_classroom_spring_voxel_resistance_from_129(
    clip_id,
    *,
    movement_orbit_deg=0.0,
):
    """From ch6_129 pre-spring zoom: fade cloud → zoom NLL voxels → spring with local clearing."""
    pose = _ch6_matrix_contrast_cloud()
    info = pose["info"]
    mu = np.asarray(pose["mu"], dtype=np.float64).reshape(3)
    W = np.asarray(pose["W"], dtype=np.float64)
    axis_lim = pose["axis_lim"]
    marker_s = float(info.get("marker_s", 28))
    belief_z = info.get("belief_z_lim")
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2

    study = np.asarray(info["cs"], dtype=np.float64)
    exam = np.asarray(info["ce"], dtype=np.float64)
    y = np.asarray(info["cy"], dtype=np.float64).reshape(-1)
    panel_xlim, panel_ylim = info["xlim"], info["ylim"]
    w_class = np.asarray(info["w_knob"], dtype=np.float64).reshape(3)
    w_target = w_class.copy()

    ax_lo, ax_hi = float(axis_lim[0]), float(axis_lim[1])
    b_zoom_lo, b_zoom_hi = (float(v) for v in CH6_SPRING_ZOOM_BIAS_BOUNDS)
    zoom_bounds = _ch6_voxel_bounds_w12_between_points(mu, w_class, axis_lim, pad_frac=0.15)
    zoom_bounds = (
        zoom_bounds[0], zoom_bounds[1], zoom_bounds[2], zoom_bounds[3],
        b_zoom_lo, b_zoom_hi,
    )
    w12_span = float(zoom_bounds[1]) - float(zoom_bounds[0])
    # Same cell count per axis as ch6_127 → voxel edge lengths scale with axis spans.
    voxel_cache = _ch6_nll_voxel_cache(study, exam, y, bounds=zoom_bounds)

    ct_elev = ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV")))
    azim_end = float(_g("CH3_LIK_W12_CT_AZIM")) + 720.0

    Xd = ch6_design(study, exam)
    H = ch6_observed_information(mu, Xd, y, ridge=CH6_RIDGE)
    newton_dir_mu = _ch6_newton_ascent_dir(mu, Xd, y)
    grad_span = 0.052
    grad_3d_scale = 0.20 * w12_span
    grad_3d_axis_lim = (zoom_bounds[0], zoom_bounds[1])

    dim_mask = np.zeros(len(W), dtype=bool)

    def _pair_kw(*, dim_u=1.0, mu_u=1.0, class_u=1.0):
        du = float(dim_u)
        return dict(
            marker_bright_mask=dim_mask if du > 1e-4 else None,
            marker_dim_scale=0.16 if du > 1e-4 else 1.0,
            highlight_mu_u=float(mu_u),
            highlight_classroom_u=float(class_u),
            highlight_classroom_color="#111111",
            show_center=True,
        )

    def _callouts(avg_u=0.0, class_u=0.0):
        out = []
        au, cu = float(avg_u), float(class_u)
        if au > 1e-4:
            out.append(dict(
                point=mu, label="average",
                color=CH6_VARIANCE_RED, u=au,
                label_placement="bottom_outside",
            ))
        if cu > 1e-4:
            out.append(dict(
                point=w_class, label="this classroom",
                color=CH6_VARIANCE_RED, u=cu,
                label_placement="top_outside",
            ))
        return out or None

    def _zoom_base(**extra):
        base = dict(
            population_fade_u=1.0,
            marker_alpha_scale=0.0,
            highlight_classroom_w=w_class,
            ch6_pair_line_u=1.0,
            ch6_pair_line_alpha=1.0,
            voxel_view_bounds=zoom_bounds,
            ch6_callouts=_callouts(avg_u=1.0, class_u=1.0),
            **_pair_kw(mu_u=1.0, class_u=1.0),
            nll_voxel_cache=voxel_cache,
            nll_voxel_sweep_u=1.0,
            nll_voxel_alpha_u=1.0,
            nll_voxel_uniform_alpha=float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA),
            nll_grad_3d_u=0.0,
            highlight_mu_w=None,
        )
        base.update(extra)
        return base

    def _emit(**kw):
        base = dict(
            xlim=panel_xlim,
            ylim=panel_ylim,
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=W,
            mu=mu,
            axis_lim=axis_lim,
            revealed_mask=np.ones(len(W), dtype=bool),
            grey_u=0.0,
            show_center=False,
            view_elev=ct_elev,
            view_azim=azim_end,
            knob_w=kw.pop("knob_w", w_class),
            ghost_fade_u=0.0,
            marker_s=marker_s,
            belief_z_lim=belief_z,
            matrix_fade_u=0.0,
            mu_threshold_2d_color=CH6_VARIANCE_RED,
            mu_threshold_2d_lw=3.6,
            ghost_threshold_2d_color=CH6_VARIANCE_RED,
            grad_span_frac=grad_span,
            nll_grad_3d=newton_dir_mu,
            nll_grad_3d_w=mu,
            nll_grad_3d_scale=grad_3d_scale,
            nll_grad_3d_axis_lim=grad_3d_axis_lim,
            nll_grad_3d_is_direction=True,
        )
        base.update(_zoom_base())
        base.update(kw)
        return _finish(_frame_population_variance_duo(**base), clip_id)

    frames: list = []

    frames.extend(_hold(_emit(
        population_fade_u=1.0,
        marker_alpha_scale=1.0,
        nll_voxel_sweep_u=0.0,
        **_pair_kw(),
    ), n_hold))

    n_cloud_out = _draft_short(32, 12)
    for k in range(n_cloud_out):
        u = smooth(float(k) / max(n_cloud_out - 1, 1))
        frames.append(_emit(
            population_fade_u=1.0,
            marker_alpha_scale=1.0 - u,
            nll_voxel_sweep_u=0.0,
            **_pair_kw(dim_u=1.0 - u, mu_u=1.0, class_u=1.0),
        ))

    n_vox_sweep = _draft_short(48, 18)
    for k in range(n_vox_sweep):
        u = smooth(float(k) / max(n_vox_sweep - 1, 1))
        frames.append(_emit(
            marker_alpha_scale=0.0,
            nll_voxel_sweep_u=u,
            **_pair_kw(mu_u=1.0, class_u=1.0),
        ))
    frames.extend(_hold(_emit(
        nll_voxel_sweep_u=1.0,
        **_pair_kw(),
    ), max(1, n_hold // 2)))

    n_grad3d = _draft_short(28, 10)
    for k in range(n_grad3d):
        u = smooth(float(k) / max(n_grad3d - 1, 1))
        frames.append(_emit(
            nll_grad_3d_u=u,
            **_pair_kw(),
        ))
    frames.extend(_hold(frames[-1], max(1, n_hold // 3)))

    n_dist_out = _draft_short(20, 8)
    for k in range(n_dist_out):
        u = smooth(float(k) / max(n_dist_out - 1, 1))
        frames.append(_emit(
            mu_threshold_2d=mu,
            mu_threshold_2d_u=u,
            knob_w=mu,
            show_2d_grads=u > 0.35,
            grad_w_live=mu,
            nll_grad_3d_u=1.0,
            **_pair_kw(mu_u=1.0, class_u=1.0),
        ))
    frames.extend(_hold(_emit(
        mu_threshold_2d=mu,
        mu_threshold_2d_u=1.0,
        knob_w=mu,
        show_2d_grads=True,
        grad_w_live=mu,
        nll_grad_3d_u=1.0,
        **_pair_kw(),
    ), n_hold))

    spring_path = _hessian_restore_path(
        mu[:2], w_target[:2], H[:2, :2],
        n_frames=_draft_short(44, 18),
        damp=1.15,
        stiffness=5.0,
    )
    newton_path_3d = _ch6_damped_newton_path_3d(
        mu, w_target, Xd, y,
        n_frames=len(spring_path),
        axis_lim=axis_lim,
        pull_gain=0.55,
    )
    newton_path_3d[-1] = w_target.copy()

    orbit_deg = float(movement_orbit_deg)
    n_spring_move = max(len(spring_path) - 1, 1)
    for i in range(1, len(spring_path)):
        t = float(i) / max(len(spring_path) - 1, 1)
        xy = spring_path[i]
        w_walk = np.asarray(newton_path_3d[i], dtype=np.float64).reshape(3).copy()
        if i == len(spring_path) - 1:
            w_walk = w_target.copy()
        ww_line = mu.copy()
        ww_line[0], ww_line[1] = float(xy[0]), float(xy[1])
        ww_line[2] = (1.0 - t) * float(mu[2]) + t * float(w_target[2])
        if i == len(spring_path) - 1:
            ww_line = w_target.copy()
        newton_dir = _ch6_newton_ascent_dir(w_walk, Xd, y)
        opaque_u = float(CH6_NLL_VOXEL_WALKER_OPAQUE_MAX) * smooth(0.2 + 0.8 * t)
        move_kw = dict(
            population_fade_u=1.0,
            highlight_classroom_w=w_class,
            mu_threshold_2d=mu,
            mu_threshold_2d_u=1.0,
            ghost_threshold_2d=mu,
            ghost_threshold_2d_u=1.0,
            classroom_threshold_2d=ww_line,
            classroom_threshold_2d_u=1.0,
            knob_w=ww_line,
            show_2d_grads=True,
            grad_w_live=ww_line,
            nll_grad_3d=newton_dir,
            nll_grad_3d_w=w_walk,
            nll_grad_3d_u=max(0.12, 1.0 - t * 0.82),
            nll_grad_3d_is_direction=True,
            ch6_callouts=_callouts(avg_u=1.0, class_u=1.0),
            **_pair_kw(mu_u=1.0, class_u=1.0),
            highlight_mu_w=w_walk,
            nll_voxel_walker_w=w_walk,
            nll_voxel_walker_opaque_u=opaque_u,
        )
        if orbit_deg > 1e-6:
            orbit_u = smooth(float(i - 1) / max(n_spring_move - 1, 1))
            move_kw["view_azim"] = _ch6_orbit_view_azim(
                azim_end, move_u=orbit_u, orbit_deg=orbit_deg,
            )
        frames.append(_emit(**move_kw))

    end_kw = dict(
        population_fade_u=1.0,
        highlight_classroom_w=w_class,
        mu_threshold_2d=mu,
        mu_threshold_2d_u=1.0,
        ghost_threshold_2d=mu,
        ghost_threshold_2d_u=1.0,
        classroom_threshold_2d=w_target,
        classroom_threshold_2d_u=1.0,
        knob_w=w_target,
        show_2d_grads=True,
        grad_w_live=w_target,
        nll_grad_3d_u=0.0,
        ch6_callouts=_callouts(avg_u=1.0, class_u=1.0),
        **_pair_kw(mu_u=1.0, class_u=1.0),
        highlight_mu_w=w_target,
    )
    if orbit_deg > 1e-6 and n_hold > 0:
        for k in range(n_hold):
            ru = smooth(float(k) / max(n_hold - 1, 1))
            frames.append(_emit(
                **end_kw,
                view_azim=_ch6_orbit_view_azim(
                    azim_end, return_u=ru, orbit_deg=orbit_deg,
                ),
            ))
    else:
        frames.extend(_hold(_emit(**end_kw), n_hold))
    return frames


def build_ch6_133_population_n100_classroom_spring_voxel_resistance_from_129(clip_id):
    """Zoomed pair → fade cloud → NLL voxels → spring push with local voxel clearing."""
    return _build_ch6_133_population_n100_classroom_spring_voxel_resistance_from_129(clip_id)


# ---------------------------------------------------------------------------
# ch6_141–143 — movement / covariance story (tutorial rails + voxel average)
# ---------------------------------------------------------------------------

CH6_MOVEMENT_BOTTOM_TITLE = "Key idea"
CH6_MOVEMENT_RIGHT_TITLE = "Reading the landscape"
CH6_MOVEMENT_SANDWICH_BOTTOM_TITLE = "Restoring Pull"
CH6_MOVEMENT_SANDWICH_RIGHT_TITLE = "Shared curvature"
CH6_SANDWICH_ESTIMATOR_BOTTOM_TITLE = "Sandwich estimator"
CH6_SANDWICH_ESTIMATOR_RIGHT_TITLE = "Plug-in pieces"
CH6_MOVEMENT_SHELL_KEY = "ch6_movement_shell_v1"
CH6_SANDWICH_SHELL_KEY = "ch6_sandwich_shell_v10"
CH6_SANDWICH_ESTIMATOR_SHELL_KEY = "ch6_sandwich_estimator_v2"

_CH6_MOVEMENT_COMPOSERS: dict[str, Any] = {}


def _ch6_movement_composer():
    """Tutorial layout tuned for ch6_141–143 (title fit, spacing, no clipping)."""
    if "movement" not in _CH6_MOVEMENT_COMPOSERS:
        from dataclasses import replace

        from ch4_layout import (
            CH4_CANVAS_H_IN,
            CH4_EXPORT,
            CH4_LAYOUT,
            CH4_RIGHT_TEXT_SHIFT_IN,
            make_composer,
        )

        lift_frac = 0.9 / float(CH4_CANVAS_H_IN)
        layout = replace(
            CH4_LAYOUT,
            right_section_drop_frac=CH4_LAYOUT.right_section_drop_frac + 0.024,
            right_rail_lower_frac=max(
                0.0, CH4_LAYOUT.right_rail_lower_frac - lift_frac * 0.45,
            ),
            right_text_shift_in=max(0.0, CH4_RIGHT_TEXT_SHIFT_IN - 0.38),
            right_row_gap_frac=CH4_LAYOUT.right_row_gap_frac * 0.20,
            bottom_text_drop_frac=CH4_LAYOUT.bottom_text_drop_frac + 0.034,
            bottom_content_lift_frac=-0.016,
        )
        _CH6_MOVEMENT_COMPOSERS["movement"] = make_composer(
            "classic_light", export=CH4_EXPORT, layout=layout,
        )
    return _CH6_MOVEMENT_COMPOSERS["movement"]


def _ch6_movement_shell_key(
    right_blocks,
    *,
    sandwich: bool = False,
    stage: int | None = None,
) -> str:
    base = CH6_SANDWICH_SHELL_KEY if sandwich else CH6_MOVEMENT_SHELL_KEY
    key = f"{base}:r{len(list(right_blocks or []))}"
    if stage is not None:
        key = f"{key}:s{int(stage)}"
    return key


def _ch6_compose_tutorial_frame(
    plot_img,
    *,
    layout_u=1.0,
    panel_u=1.0,
    title_u=1.0,
    write_u=1.0,
    bottom_blocks=None,
    right_blocks=None,
    bottom_prog=None,
    right_prog=None,
    bottom_title="",
    right_title="",
    plot_start_rect=None,
    shell_cache_key=None,
    right_title_single_line=False,
    composer=None,
):
    compose = _g("ch4_compose_tutorial_frame")
    po = {}
    if bottom_blocks is not None and bottom_prog is not None:
        po["bottom"] = bottom_prog
    if right_blocks is not None and right_prog is not None:
        po["right"] = right_prog
    kw = dict(
        plot_img=plot_img,
        right_blocks=list(right_blocks or []),
        bottom_blocks=list(bottom_blocks or []),
        corner_blocks=[],
        right_title=str(right_title),
        bottom_title=str(bottom_title),
        corner_title="",
        layout_u=float(layout_u),
        panel_u=float(panel_u),
        title_write_progress=float(title_u),
        write_progress=float(write_u),
        theme="classic_light",
        right_title_single_line=bool(right_title_single_line),
        composer=composer or _ch6_movement_composer(),
    )
    if plot_start_rect is not None:
        kw["plot_start_rect"] = plot_start_rect
    elif float(layout_u) < 1.0 - 1e-9:
        kw["plot_start_rect"] = _g("CH4_LIK_PLOT_START_RECT")
    if shell_cache_key is not None:
        kw["shell_cache_key"] = str(shell_cache_key)
    if po:
        kw["progress_override"] = po
    return compose(**kw)


def _ch6_handwrite_block(text, *, fs=26.0, weight=0.0, align="left", mathtext_lines=None, **kw):
    block = _g("_ch4_formula_hand_block")(
        str(text),
        weight=float(weight),
        align=str(align),
        text_x_frac=0.04,
        block_fs=float(fs),
        line_dy_pt=22.0,
        bold_lhs=False,
        pt_units=True,
        top_pad_pt=0.0,
        text_y_inset_pt=4.0,
        force_wrap=True,
        mathtext_max_line_frac=0.90,
        block_fs_min=14.0,
    )
    block.update(kw)
    if mathtext_lines is not None:
        block["mathtext_lines"] = [str(ln) for ln in mathtext_lines]
        if "mathtext_fs" not in block:
            block["mathtext_fs"] = float(fs * 0.92)
    return block


def _ch6_block_write_prog(blocks, slot_prog: dict[int, float]):
    """Per-block line progress for tutorial rails."""
    import handwrite_tutorial as hw

    style = _ch6_movement_composer().handwrite_style()
    out: dict[int, dict] = {}
    for bi, block in enumerate(blocks):
        p = float(slot_prog.get(bi, 1.0))
        bp = hw.block_write_progress([block], p, style=style)
        if 0 in bp:
            out[bi] = bp[0]
    return out


def _ch6_block_write_prog_uniform(blocks, u: float):
    return _ch6_block_write_prog(blocks, {bi: float(u) for bi in range(len(blocks))})


def _ch6_block_write_prog_frozen_lead(blocks, u: float, *, frozen_lead: int = 1):
    """Animate trailing blocks only; lead blocks stay fully written."""
    n = len(blocks)
    frozen_lead = min(int(frozen_lead), n)
    return _ch6_block_write_prog(
        blocks,
        {bi: (1.0 if bi < frozen_lead else float(u)) for bi in range(n)},
    )


def _ch6_prose_block(text, *, fs=19.0, label=None, pre_gap_pt=0.0):
    kw = dict(
        fs=float(fs),
        weight=0.0,
        line_dy_pt=19.0,
        text_x_frac=0.03,
        mathtext_max_line_frac=0.88,
    )
    if label is not None:
        kw.update(label=str(label), label_fs=22.0, label_gap_pt=10.0, pre_gap_pt=float(pre_gap_pt))
    return _ch6_handwrite_block(str(text), **kw)


def _ch6_bottom_formula_block(stage: int):
    """Bottom-rail formula stages; movement is H⁻¹∇NLL (inverse Hessian left-multiples gradient)."""
    drop_pt = 36.0
    common = dict(
        align="center",
        text_x_frac=0.5,
        text_y_inset_pt=drop_pt,
        weight=0.34,
        fs=21.0,
        line_dy_pt=24.0,
        mathtext_max_line_frac=0.94,
    )
    stages = [
        "Total Movement from the average ≈ Current Strength / Current Decay",
        "Total Movement from the average ≈ Gradient at the average line / Current Decay",
        "Total Movement from the average ≈ Gradient at the average line / Curvature at the average line",
    ]
    if int(stage) < 3:
        return [_ch6_handwrite_block(stages[int(stage)], **common)]
    return [_ch6_handwrite_block(
        "",
        **common,
        mathtext_lines=[
            r"$\Delta w \;\approx\; H_{\mathrm{class}}^{-1}\,"
            r"\nabla\mathrm{NLL}_{\mathrm{class}}(\mu)$",
        ],
        mathtext_fs=27.0,
    )]


def _ch6_movement_story_pack():
    """Shared scene for ch6_141–143 (same classroom / cloud as ch6_133)."""
    pose = _ch6_matrix_contrast_cloud()
    info = pose["info"]
    mu = np.asarray(pose["mu"], dtype=np.float64).reshape(3)
    W = np.asarray(pose["W"], dtype=np.float64)
    study = np.asarray(info["cs"], dtype=np.float64)
    exam = np.asarray(info["ce"], dtype=np.float64)
    y = np.asarray(info["cy"], dtype=np.float64).reshape(-1)
    w_class = np.asarray(info["w_knob"], dtype=np.float64).reshape(3)
    axis_lim = pose["axis_lim"]
    ax_lo, ax_hi = float(axis_lim[0]), float(axis_lim[1])
    full_bounds = (ax_lo, ax_hi, ax_lo, ax_hi, ax_lo, ax_hi)
    b_zoom_lo, b_zoom_hi = (float(v) for v in CH6_SPRING_ZOOM_BIAS_BOUNDS)
    zoom_bounds = _ch6_voxel_bounds_w12_between_points(mu, w_class, axis_lim, pad_frac=0.15)
    zoom_bounds = (
        zoom_bounds[0], zoom_bounds[1], zoom_bounds[2], zoom_bounds[3],
        b_zoom_lo, b_zoom_hi,
    )
    # Match ch6_133: fixed cells-per-axis at the zoom window (no cubic override).
    voxel_class = _ch6_nll_voxel_cache(study, exam, y, bounds=zoom_bounds)
    Xd = ch6_design(study, exam)
    H_avg = ch6_observed_information(mu, Xd, y, ridge=CH6_RIDGE)
    newton_dir_mu = _ch6_newton_ascent_dir(mu, Xd, y)
    ct_elev = ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV")))
    azim_end = float(_g("CH3_LIK_W12_CT_AZIM")) + 720.0
    w12_span = float(zoom_bounds[1]) - float(zoom_bounds[0])
    grad_3d_scale = 0.20 * w12_span
    grad_3d_axis_lim = (zoom_bounds[0], zoom_bounds[1])
    pack = _ch6_114_end_pack()
    st = pack["st"]
    return dict(
        pose=pose,
        info=info,
        mu=mu,
        W=W,
        study=study,
        exam=exam,
        y=y,
        w_class=w_class,
        axis_lim=axis_lim,
        zoom_bounds=zoom_bounds,
        full_bounds=full_bounds,
        voxel_class=voxel_class,
        Xd=Xd,
        H_avg=H_avg,
        newton_dir_mu=newton_dir_mu,
        ct_elev=ct_elev,
        azim_end=azim_end,
        grad_3d_scale=grad_3d_scale,
        grad_3d_axis_lim=grad_3d_axis_lim,
        panel_xlim=info["xlim"],
        panel_ylim=info["ylim"],
        marker_s=float(info.get("marker_s", 28)),
        belief_z=info.get("belief_z_lim"),
        pop_s=st["pop_s"],
        pop_e=st["pop_e"],
        pop_y=st["pop_y"],
    )


def _ch6_movement_plot_raw(scene, **kw):
    """Raw duo raster (no tutorial compose) for ch6_141–143."""
    s = scene
    base = dict(
        xlim=s["panel_xlim"],
        ylim=s["panel_ylim"],
        base_study=s["info"]["cs"],
        base_exam=s["info"]["ce"],
        base_y=s["info"]["cy"],
        W=s["W"],
        mu=s["mu"],
        axis_lim=s["axis_lim"],
        revealed_mask=np.ones(len(s["W"]), dtype=bool),
        grey_u=0.0,
        show_center=False,
        view_elev=s["ct_elev"],
        view_azim=s["azim_end"],
        knob_w=kw.pop("knob_w", s["w_class"]),
        ghost_fade_u=0.0,
        marker_s=s["marker_s"],
        belief_z_lim=s["belief_z"],
        matrix_fade_u=0.0,
        mu_threshold_2d_color=CH6_VARIANCE_RED,
        mu_threshold_2d_lw=3.6,
        ghost_threshold_2d_color=CH6_VARIANCE_RED,
        grad_span_frac=0.052,
        nll_grad_3d=s["newton_dir_mu"],
        nll_grad_3d_w=s["mu"],
        nll_grad_3d_scale=s["grad_3d_scale"],
        nll_grad_3d_axis_lim=s["grad_3d_axis_lim"],
        nll_grad_3d_is_direction=True,
    )
    base.update(kw)
    return _frame_population_variance_duo(**base)


def _ch6_movement_refined_roster(scene):
    """Demo classrooms + population draws — shared by ch6_142 zoom landscape and ch6_143."""
    from ch6_frequentist import _CH3_DRAFT

    s = scene
    n_avg = 20 if _CH3_DRAFT else 60
    rng = np.random.default_rng(142)
    demo_idx = _ch6_pick_classroom_demo_indices(s["W"], s["w_class"], n=5, seed=141)
    classrooms = _ch6_classroom_roster_for_indices(scene, demo_idx)
    big_roster = list(classrooms)
    while len(big_roster) < n_avg:
        cs, ce, cy, _ = ch6_draw_classroom_from_population(
            s["pop_s"], s["pop_e"], s["pop_y"], 100, rng=rng,
        )
        big_roster.append((cs, ce, cy))
    return big_roster


def _ch6_movement_zoom_voxel_cache(scene, roster=None):
    """Refined multi-classroom average at the ch6_133 zoom window (ch6_142 reverse-push landscape)."""
    s = scene
    roster = roster if roster is not None else _ch6_movement_refined_roster(scene)
    return _ch6_nll_voxel_cache_average(
        roster, bounds=s["zoom_bounds"], mu=s["mu"], H=s["H_avg"], cache_kw={},
    )


def _ch6_movement_pre_move_kw(scene):
    """ch6_133 hold: voxels + gradient, before spring / thresholds."""
    s = scene
    return dict(
        population_fade_u=1.0,
        marker_alpha_scale=0.0,
        highlight_classroom_w=s["w_class"],
        ch6_pair_line_u=1.0,
        ch6_pair_line_alpha=1.0,
        voxel_view_bounds=s["zoom_bounds"],
        nll_voxel_cache=s["voxel_class"],
        nll_voxel_sweep_u=1.0,
        nll_voxel_alpha_u=1.0,
        nll_voxel_uniform_alpha=float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA),
        nll_grad_3d_u=1.0,
        highlight_mu_u=1.0,
        highlight_classroom_u=1.0,
        highlight_classroom_color="#111111",
        highlight_mu_w=s["mu"],
        ch6_callouts=[
            dict(point=s["mu"], label="average", color=CH6_VARIANCE_RED, u=1.0, label_placement="bottom_outside"),
            dict(point=s["w_class"], label="this classroom", color=CH6_VARIANCE_RED, u=1.0, label_placement="top_outside"),
        ],
    )


def _ch6_movement_bottom_blocks(stage: int):
    return _ch6_bottom_formula_block(stage)


def _ch6_movement_right_blocks(stage: int):
    """Right-rail prose for ch6_141 (Goal subsection after stage 0)."""
    landscape = _ch6_prose_block(
        "The landscape shown is the NLL\nof this particular classroom.",
    )
    si = int(np.clip(int(stage), 0, 4))
    if si == 0:
        return [landscape]
    if si == 1:
        return [
            landscape,
            _ch6_prose_block(
                "understand the covariance\nof the best lines.",
                label="Goal",
                pre_gap_pt=16.0,
            ),
        ]
    if si == 2:
        return [
            landscape,
            _ch6_prose_block(
                "understand the covariance\nof the best lines.\n\n"
                "We study what drives departure\nfrom the average line.\n\n"
                "Covariance of that departure\nis our target.",
                label="Goal",
                pre_gap_pt=16.0,
            ),
        ]
    if si == 3:
        return [
            landscape,
            _ch6_prose_block(
                "understand the covariance\nof the best lines.\n\n"
                "The Hessian depends on the\nclassroom — and each gradient\n"
                "depends on the classroom.\n\nThat makes this tricky.",
                label="Goal",
                pre_gap_pt=16.0,
            ),
        ]
    return [
        landscape,
        _ch6_prose_block(
            "understand the covariance\nof the best lines.\n\n"
            "Computing the covariance is\nchallenging: every classroom\n"
            "has a different push and a\ndifferent decay.",
            label="Goal",
            pre_gap_pt=16.0,
        ),
    ]


def _ch6_sandwich_avg_grad_at_class():
    """Gradient of the average landscape, evaluated at this classroom (ch6_143)."""
    return r"\nabla\overline{\mathrm{NLL}}(w_{\mathrm{class}})"


def _ch6_sandwich_bottom_blocks(stage: int):
    """Bottom-rail formulas for ch6_143 — match ch6_141 equation sizing."""
    drop_pt = 36.0
    math_fs = 27.0
    math_fs_sm = 24.0
    grad = _ch6_sandwich_avg_grad_at_class()
    common = dict(
        align="center",
        text_x_frac=0.5,
        text_y_inset_pt=drop_pt,
        weight=0.34,
        fs=21.0,
        line_dy_pt=24.0,
        mathtext_max_line_frac=0.94,
        mathtext_min_fs=20.0,
    )
    si = int(stage)
    if si == 0:
        return [_ch6_handwrite_block(
            "Restoring pull: decay = Hessian of the average landscape "
            "(same for every classroom)",
            **common,
        )]
    if si == 1:
        return [_ch6_handwrite_block(
            "",
            **common,
            mathtext_lines=[
                r"$\text{Total movement} \approx H^{-1}\," + grad + r"$",
            ],
            mathtext_fs=math_fs,
        )]
    if si == 2:
        return [_ch6_handwrite_block(
            "",
            **common,
            mathtext_lines=[
                r"$\mathrm{Cov}(\text{Total movement}) ="
                r"\mathrm{Cov}\!\left(H^{-1}\," + grad + r"\right)$",
            ],
            mathtext_fs=math_fs_sm,
        )]
    if si == 3:
        return [_ch6_handwrite_block(
            "",
            **common,
            mathtext_lines=[
                r"$\mathrm{Cov}(\text{Total movement}) ="
                r"H^{-1}\,\mathrm{Cov}\!\left(" + grad + r"\right)\,H^{-1}$",
            ],
            mathtext_fs=math_fs_sm,
        )]
    if si == 4:
        return [_ch6_handwrite_block(
            "",
            **common,
            mathtext_lines=[
                r"$\mathrm{Cov}(\text{Total movement}) = H^{-1}\,J\,H^{-1}$",
            ],
            mathtext_fs=math_fs,
        )]
    raise ValueError(f"unknown sandwich bottom stage {si}")


def _ch6_sandwich_right_blocks(stage: int):
    grad = _ch6_sandwich_avg_grad_at_class()
    shared = _ch6_prose_block(
        "The average landscape's\n"
        "curvature is shared among\n"
        "all classrooms.",
    )
    if int(stage) < 4:
        return [shared]
    return [
        shared,
        _ch6_handwrite_block(
            "",
            fs=19.0,
            weight=0.0,
            line_dy_pt=22.0,
            text_x_frac=0.03,
            mathtext_max_line_frac=0.88,
            pre_gap_pt=14.0,
            label="Notation",
            label_fs=22.0,
            label_gap_pt=10.0,
            mathtext_lines=[r"$J = \mathrm{Cov}(" + grad + r")$"],
            mathtext_fs=24.0,
        ),
    ]


def _ch6_sandwich_estimator_bottom_blocks(stage: int):
    """Bottom rail: true sandwich → Ĥ_class and Ĵ substitutions (ch6_160)."""
    drop_pt = 36.0
    math_fs_sm = 23.0
    grad = _ch6_sandwich_avg_grad_at_class()
    cov_g = r"\mathrm{Cov}\!\left(" + grad + r"\right)"
    h_hat = r"\widehat{H}_{\mathrm{class}}"
    common = dict(
        align="center",
        text_x_frac=0.5,
        text_y_inset_pt=drop_pt,
        weight=0.34,
        fs=21.0,
        line_dy_pt=24.0,
        mathtext_max_line_frac=0.96,
        mathtext_min_fs=19.0,
    )
    si = int(stage)
    if si == 0:
        return [_ch6_handwrite_block(
            "",
            **common,
            mathtext_lines=[
                r"$\mathrm{Cov}(\Delta w) = H^{-1}\," + cov_g + r"\,H^{-1}$",
            ],
            mathtext_fs=math_fs_sm,
        )]
    if si == 1:
        return [_ch6_handwrite_block(
            "",
            **common,
            mathtext_lines=[
                r"$\mathrm{Cov}(\Delta w) = "
                + h_hat + r"^{-1}\," + cov_g + r"\," + h_hat + r"^{-1}$",
            ],
            mathtext_fs=math_fs_sm,
        )]
    if si == 2:
        return [_ch6_handwrite_block(
            "",
            **common,
            mathtext_lines=[
                r"$\mathrm{Cov}(\Delta w) = "
                + h_hat + r"^{-1}\,\widehat{J}\," + h_hat + r"^{-1}$",
            ],
            mathtext_fs=math_fs_sm,
        )]
    if si == 3:
        return [_ch6_handwrite_block(
            "",
            **common,
            mathtext_lines=[
                r"$\mathrm{Cov}(\Delta w) \approx "
                + h_hat + r"^{-1}\,\widehat{J}\," + h_hat + r"^{-1}$",
            ],
            mathtext_fs=math_fs_sm,
        )]
    raise ValueError(f"unknown sandwich estimator bottom stage {si}")


def _ch6_sandwich_estimator_right_blocks(stage: int):
    """Minimal right-rail labels for each plug-in estimate."""
    si = int(stage)
    note = dict(
        fs=19.0,
        weight=0.0,
        line_dy_pt=22.0,
        text_x_frac=0.03,
        mathtext_max_line_frac=0.90,
        pre_gap_pt=10.0,
        label_fs=21.0,
        label_gap_pt=8.0,
    )
    if si == 0:
        return _ch6_sandwich_right_blocks(0)
    if si == 1:
        return [_ch6_handwrite_block(
            "",
            **note,
            label="decay",
            mathtext_lines=[r"$\widehat{H}_{\mathrm{class}}$"],
            mathtext_fs=26.0,
        )]
    if si == 2:
        return [_ch6_handwrite_block(
            "",
            **note,
            label="spread",
            mathtext_lines=[r"$\widehat{J}=\mathrm{Cov}(\nabla_i)$"],
            mathtext_fs=24.0,
        )]
    if si == 3:
        note2 = dict(note)
        note2["pre_gap_pt"] = 18.0
        return [
            _ch6_handwrite_block(
                "",
                **note,
                label="decay",
                mathtext_lines=[r"$\widehat{H}_{\mathrm{class}}$"],
                mathtext_fs=24.0,
            ),
            _ch6_handwrite_block(
                "",
                **note2,
                label="spread",
                mathtext_lines=[r"$\widehat{J}=\mathrm{Cov}(\nabla_i)$"],
                mathtext_fs=22.0,
            ),
        ]
    raise ValueError(f"unknown sandwich estimator right stage {si}")


def _ch6_2d_grad_away_from_line_quiver(
    study,
    exam,
    contrib,
    w,
    *,
    span_frac=0.16,
    xlim=(-3.0, 3.0),
    ylim=(-3.0, 3.0),
    compact=True,
):
    """Per-point away-from-line quiver UV — mirrors ``_draw_point_grad_quivers_away_from_line_2d``."""
    study = np.asarray(study, dtype=np.float64).reshape(-1)
    exam = np.asarray(exam, dtype=np.float64).reshape(-1)
    G = np.asarray(contrib, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(3)
    n = len(study)
    if G.ndim != 2 or n == 0:
        return None
    a, b, c = float(w[0]), float(w[1]), float(w[2])
    den = a * a + b * b
    if den < 1e-12:
        return None
    z = a * study + b * exam + c
    t = -z / den
    foot_s = study + t * a
    foot_e = exam + t * b
    away_s = study - foot_s
    away_e = exam - foot_e
    dist = np.hypot(away_s, away_e)
    mask = dist > 1e-4
    if not np.any(mask):
        return None
    G2 = G[:, :2]
    proj = np.zeros(n, dtype=np.float64)
    d_hat = np.zeros((n, 2), dtype=np.float64)
    for i in np.where(mask)[0]:
        d_hat[i] = away_s[i] / dist[i], away_e[i] / dist[i]
        proj[i] = abs(float(G2[i] @ d_hat[i]))
    xlo, xhi = float(xlim[0]), float(xlim[1])
    ylo, yhi = float(ylim[0]), float(ylim[1])
    in_ax = (
        (study >= xlo) & (study <= xhi) & (exam >= ylo) & (exam <= yhi)
    )
    mask = mask & in_ax
    if not np.any(mask):
        return None
    peak_ref = float(np.max(proj[mask])) + 1e-12
    target = float(span_frac) * min(xhi - xlo, yhi - ylo)
    if compact:
        target *= 0.72
    U = np.zeros(n, dtype=np.float64)
    V = np.zeros(n, dtype=np.float64)
    for i in np.where(mask)[0]:
        mag = float(proj[i])
        if mag < 1e-9:
            continue
        length = min(mag / peak_ref, 1.0) * target
        if length < 0.05 * target:
            continue
        U[i] = length * float(d_hat[i, 0])
        V[i] = length * float(d_hat[i, 1])
    draw_ix = np.where((np.abs(U) + np.abs(V)) > 0)[0]
    if len(draw_ix) == 0:
        return None
    idx = int(draw_ix[int(np.argmax(proj[draw_ix]))])
    return int(idx), float(U[idx]), float(V[idx])


def _ch6_sandwich_160_callouts(
    mu,
    w_class,
    *,
    avg_u=0.0,
    class_u=0.0,
    grad_tip=None,
    grad_u=0.0,
    hess_tip=None,
    hess_u=0.0,
):
    """ch6_160 callouts — fixed rim slots, clear of 2D panel and tutorial rails."""
    out: list = []
    au, cu = float(avg_u), float(class_u)
    if au > 1e-4:
        out.append(dict(
            point=mu, label="average",
            color=CH6_TRUE_LABEL_COLOR, u=au,
            label_anchor="sandwich_avg_tr",
        ))
    if cu > 1e-4:
        out.append(dict(
            point=w_class, label="this classroom",
            color="#111111", u=cu,
            label_anchor="sandwich_class_tl",
        ))
    if grad_tip is not None and float(grad_u) > 1e-4:
        out.append(dict(
            point=grad_tip, label=r"$\nabla\overline{\mathrm{NLL}}$",
            color=CH6_TRUE_LABEL_COLOR, u=float(grad_u),
            label_anchor="sandwich_grad",
        ))
    if hess_tip is not None and float(hess_u) > 1e-4:
        out.append(dict(
            point=hess_tip, label=r"$\widehat{H}_{\mathrm{class}}$",
            color="#111111", u=float(hess_u),
            label_anchor="sandwich_hess",
        ))
    return out or None


def _ch6_sandwich_160_largest_in_bounds_quiver(study, exam, y, w, *, xlim, ylim, span_frac):
    """Largest in-bounds per-student quiver (idx, du, dv)."""
    G = ch6_point_nll_grad_contrib(w, study, exam, y)
    return _ch6_2d_grad_away_from_line_quiver(
        study, exam, G, w,
        span_frac=float(span_frac), xlim=xlim, ylim=ylim, compact=True,
    )


def _ch6_sandwich_160_data_callout(
    study, exam, y, w, *, u=1.0, span_frac=0.16, xlim=(-3.0, 3.0), ylim=(-3.0, 3.0),
):
    """2D callout at the former average slot → tip of largest in-bounds student vector."""
    if float(u) <= 1e-4 or study is None or len(study) == 0:
        return None
    hit = _ch6_sandwich_160_largest_in_bounds_quiver(
        study, exam, y, w, xlim=xlim, ylim=ylim, span_frac=float(span_frac),
    )
    if hit is None:
        return None
    idx, du, dv = hit
    lx, ly = CH6_SANDWICH_160_PER_STUDENT_LABEL_FIG
    return dict(
        xy=(float(study[idx]) + du, float(exam[idx]) + dv),
        label=r"per-student $\nabla_i$",
        color=CH6_ESTIMATOR_LABEL_COLOR,
        label_fig=(float(lx), float(ly)),
        label_ha="left",
        label_va="top",
        u=float(u),
    )


def _ch6_hessian_arrow_tip(w, dirs, lengths, *, which=0):
    """Tip of one Hessian eigen-arrow for callout placement."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    dirs = list(dirs)
    lengths = list(lengths)
    if not dirs or not lengths:
        return w
    wi = int(which)
    if wi < 0 or wi >= len(dirs):
        wi = 0
    d = np.asarray(dirs[wi], dtype=np.float64).reshape(3)
    L = float(lengths[wi])
    if len(lengths) > 1 and float(lengths[1]) > L:
        d = np.asarray(dirs[1], dtype=np.float64).reshape(3)
        L = float(lengths[1])
    return w + L * d


def _ch6_avg_grad_156_end_kw(pack, s, *, n_pts):
    """Plot kwargs matching the last frame of ch6_156."""
    last_i = len(pack["pick_idx"]) - 1
    idx = int(pack["pick_idx"][last_i])
    cs, ce, cy = pack["picked_classrooms"][last_i]
    w = pack["picked_weights"][last_i]
    mu = np.asarray(s["mu"], dtype=np.float64).reshape(3)
    vox_uniform = float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA)
    zb = tuple(pack["zoom_bounds"])
    zoom_vox_kw = _ch6_voxel_sweep_layer(
        pack["vox_avg_zoom"], zb,
        sweep_u=1.0, uniform_alpha=vox_uniform,
    )
    panel_kw = _ch6_classroom_panel_kw(cs, ce, cy)
    panel_kw.update(
        mu_threshold_2d=mu, mu_threshold_2d_u=1.0,
        ghost_threshold_2d=mu, ghost_threshold_2d_u=1.0,
        classroom_threshold_2d=w, classroom_threshold_2d_u=1.0,
        classroom_threshold_2d_color="#111111",
    )
    kw = {
        **dict(
            population_fade_u=1.0,
            marker_alpha_scale=0.90,
            grey_u=0.82,
            marker_dim_scale=0.16,
            highlight_mu_u=0.95,
            highlight_mu_w=mu,
            ch6_pair_line_u=0.80,
            highlight_classroom_u=1.0,
            highlight_classroom_color="#111111",
            view_azim=float(s["azim_end"]),
            nll_grad_3d=pack["grad_dirs"][last_i],
            nll_grad_3d_w=w,
            nll_grad_3d_scale=pack["grad_scales"][last_i],
            nll_grad_3d_u=1.0,
            nll_grad_3d_is_direction=True,
            nll_grad_3d_color="#111111",
            nll_grad_3d_axis_lim=pack["grad_3d_axis_lim"],
        ),
        **zoom_vox_kw,
        **panel_kw,
    }
    kw.update(_ch6_black_points_kw(n_pts, [idx]))
    return kw, w, cs, ce, cy


def _ch6_nll_voxel_cache_average(classrooms, *, bounds, mu=None, H=None, cache_kw=None):
    """Mean NLL voxel grid over many (study, exam, y) classrooms."""
    classrooms = list(classrooms)
    if not classrooms:
        raise ValueError("classrooms required")
    ref = _ch6_nll_voxel_cache(
        classrooms[0][0], classrooms[0][1], classrooms[0][2],
        bounds=bounds, **dict(cache_kw or {}),
    )
    n1, n2, n3 = ref["n"]
    w1 = 0.5 * (ref["x_edges"][:-1] + ref["x_edges"][1:])
    w2 = 0.5 * (ref["y_edges"][:-1] + ref["y_edges"][1:])
    bb = 0.5 * (ref["z_edges"][:-1] + ref["z_edges"][1:])
    W1, W2, B = np.meshgrid(w1, w2, bb, indexing="ij")
    acc = np.zeros(W1.shape, dtype=np.float64)
    for cs, ce, cy in classrooms:
        nll = _g("_ch3_nll_sum_on_flat_grid")(
            np.asarray(cs, dtype=np.float64),
            np.asarray(ce, dtype=np.float64),
            np.asarray(cy, dtype=np.float64).reshape(-1),
            W1.ravel(), W2.ravel(), B.ravel(),
        ).reshape(W1.shape)
        acc += nll
    acc /= float(len(classrooms))
    g_lo, g_hi = _g("ch4_nll_global_scale")()
    rgba = np.asarray(ch4_nll_heatmap_facecolors(acc, vmin=g_lo, vmax=g_hi, alpha=1.0), dtype=np.float32)
    mu_ref = np.asarray(mu if mu is not None else ref["mu"], dtype=np.float64).reshape(3)
    H_ref = np.asarray(H if H is not None else ref["H"], dtype=np.float64)
    curv_quad = _ch6_nll_voxel_curv_quad(W1, W2, B, mu_ref, H_ref).astype(np.float32, copy=False)
    return {
        "n": (n1, n2, n3),
        "x_edges": ref["x_edges"],
        "y_edges": ref["y_edges"],
        "z_edges": ref["z_edges"],
        "checker": ref["checker"],
        "proj": ref["proj"],
        "max_proj": ref["max_proj"],
        "rgb": rgba[..., :3],
        "curv_quad": curv_quad,
        "mu": mu_ref,
        "H": H_ref,
        "bounds": ref["bounds"],
    }


def _ch6_pick_classroom_demo_indices(W, w_class, *, n=5, seed=141):
    W = np.asarray(W, dtype=np.float64)
    w_class = np.asarray(w_class, dtype=np.float64).reshape(3)
    feat = int(np.argmin(np.linalg.norm(W - w_class.reshape(1, 3), axis=1)))
    rng = np.random.default_rng(int(seed))
    pool = [i for i in range(len(W)) if i != feat]
    rng.shuffle(pool)
    picks = [feat]
    stride = max(len(W) // max(n, 1), 1)
    for j in range(n - 1):
        if j < len(pool):
            picks.append(int(pool[j]))
        else:
            picks.append(int((feat + (j + 1) * stride) % len(W)))
    out: list[int] = []
    for i in picks:
        if i not in out:
            out.append(int(i))
    while len(out) < n:
        cand = int(rng.integers(0, len(W)))
        if cand not in out:
            out.append(cand)
    return out[:n]


def _ch6_classroom_roster_for_indices(
    scene,
    indices,
    *,
    n_class: int = 100,
    rng_seed: int = 141,
    seed_from_key: str | None = None,
):
    """Featured classroom + population draws for voxel-average demo."""
    s = scene
    n_class = int(n_class)
    rng = np.random.default_rng(int(rng_seed))
    out = []
    for k, idx in enumerate(indices):
        del idx  # roster order follows ``indices``; draws are population-based
        if k == 0 and seed_from_key is not None:
            tgt_s, tgt_e, tgt_y = ch5_unpack_dataset(str(seed_from_key))
            cs, ce, cy, _ = ch6_match_roster_from_population(
                s["pop_s"], s["pop_e"], s["pop_y"],
                np.asarray(tgt_s, dtype=np.float64),
                np.asarray(tgt_e, dtype=np.float64),
                np.asarray(tgt_y, dtype=np.float64),
            )
            out.append((cs, ce, cy))
        elif k == 0 and len(s["y"]) == n_class:
            out.append((s["study"], s["exam"], s["y"]))
        else:
            cs, ce, cy, _ = ch6_draw_classroom_from_population(
                s["pop_s"], s["pop_e"], s["pop_y"], n_class, rng=rng,
            )
            out.append((cs, ce, cy))
    return out


def _ch6_classroom_panel_kw(cs, ce, cy):
    """2D duo panel kwargs for one classroom roster (dataset + fitted knobs)."""
    w, _ = ch6_fit_population_classroom(cs, ce, cy)
    w = np.asarray(w, dtype=np.float64).reshape(3)
    return dict(
        base_study=np.asarray(cs, dtype=np.float64),
        base_exam=np.asarray(ce, dtype=np.float64),
        base_y=np.asarray(cy, dtype=np.float64).reshape(-1),
        knob_w=w,
        highlight_classroom_w=w,
    )


def _ch6_classroom_weights(classrooms):
    out = []
    for cs, ce, cy in classrooms:
        w, _ = ch6_fit_population_classroom(cs, ce, cy)
        out.append(np.asarray(w, dtype=np.float64).reshape(3))
    return out


def _ch6_visit_classroom_panel_kw(ki, classrooms, weights):
    """Active classroom on 2D; prior rosters + threshold lines as ghosts."""
    cs, ce, cy = classrooms[int(ki)]
    kw = _ch6_classroom_panel_kw(cs, ce, cy)
    prior = list(classrooms[: int(ki)])
    prior_ws = list(weights[: int(ki)])
    if prior:
        kw["ghost_classroom_rosters"] = prior
    if prior_ws:
        kw["ghost_ws"] = prior_ws
        kw["ghost_fade_u"] = 1.0
    return kw


def _ch6_all_classrooms_ghost_panel_kw(classrooms, weights, *, mu):
    """All visited classrooms ghosted on 2D for the final average reveal."""
    empty = np.array([], dtype=np.float64)
    return dict(
        base_study=empty,
        base_exam=empty,
        base_y=empty,
        ghost_classroom_rosters=list(classrooms),
        ghost_ws=list(weights),
        ghost_fade_u=1.0,
        knob_w=np.asarray(mu, dtype=np.float64).reshape(3),
        highlight_mu_w=np.asarray(mu, dtype=np.float64).reshape(3),
        highlight_mu_u=0.85,
        highlight_classroom_u=0.0,
    )


def _ch6_weight_in_voxel_bounds(w, bounds, *, margin=0.0):
    """True when ``w`` lies inside a 6-tuple voxel/view bounds box."""
    ww = np.asarray(w, dtype=np.float64).reshape(3)
    lo1, hi1, lo2, hi2, lob, hib = (float(v) for v in bounds)
    m = float(margin)
    return (
        lo1 - m <= float(ww[0]) <= hi1 + m
        and lo2 - m <= float(ww[1]) <= hi2 + m
        and lob - m <= float(ww[2]) <= hib + m
    )


def _ch6_avg_nll_grad_at_w(w, avg_roster):
    """Gradient of the mean classroom NLL landscape, evaluated at ``w``."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    acc = np.zeros(3, dtype=np.float64)
    for cs, ce, cy in avg_roster:
        acc += ch6_nll_grad(
            w, ch6_design(cs, ce), np.asarray(cy, dtype=np.float64).reshape(-1),
            ridge=CH6_RIDGE,
        )
    return acc / max(len(avg_roster), 1)


def _ch6_avg_grad_variation_pack(
    *,
    n_class: int = 100,
    n_pick: int = 50,
    n_demo: int = 10,
    seed: int = 141,
):
    """Precompute scene + 50-classroom avg-gradient beat for ch6_156–159."""
    from ch6_frequentist import _CH3_DRAFT

    scene = _ch6_classroom_size_workshop_scene(
        n_class=int(n_class), n_demo=int(n_demo), seed=int(seed),
    )
    pop_pack = ch6_population_param_cloud_pack(
        int(n_class), seed=int(seed), n_reel=max(int(n_class), int(CH6_N_POP_REEL)),
    )
    reel = pop_pack["reel_steps"][: int(n_class)]
    avg_roster = [
        (
            np.asarray(step["study"], dtype=np.float64),
            np.asarray(step["exam"], dtype=np.float64),
            np.asarray(step["y"], dtype=np.float64).reshape(-1),
        )
        for step in reel
    ]
    W = np.asarray(scene["W"], dtype=np.float64)
    n_cloud = min(len(W), len(reel))
    n_pick = int(min(n_pick, n_cloud))

    mu = np.asarray(scene["mu"], dtype=np.float64).reshape(3)
    axis_lim = scene["axis_lim"]
    full_bounds = scene["full_bounds"]
    w_spread = W[len(W) // 2]
    zoom_bounds = None
    in_bounds: list[int] = []
    for pad_frac in (0.18, 0.24, 0.32, 0.42):
        z12 = _ch6_voxel_bounds_w12_between_points(
            mu, w_spread, axis_lim, pad_frac=float(pad_frac),
        )
        zoom_bounds = (
            float(z12[0]), float(z12[1]), float(z12[2]), float(z12[3]),
            float(full_bounds[4]), float(full_bounds[5]),
        )
        in_bounds = [
            int(i) for i in range(n_cloud)
            if _ch6_weight_in_voxel_bounds(W[i], zoom_bounds)
        ]
        if len(in_bounds) >= n_pick:
            break
    if not in_bounds:
        in_bounds = list(range(n_cloud))
    n_pick = int(min(n_pick, len(in_bounds)))
    pick_idx = [
        int(in_bounds[int(x)])
        for x in np.linspace(0, len(in_bounds) - 1, n_pick, dtype=int)
    ]
    assert zoom_bounds is not None
    w12_span = float(zoom_bounds[1]) - float(zoom_bounds[0])
    grad_3d_scale = 0.24 * w12_span
    grad_3d_axis_lim = (zoom_bounds[0], zoom_bounds[1])

    picked_classrooms: list = []
    picked_weights: list = []
    grad_dirs: list = []
    grad_mags: list = []
    for idx in pick_idx:
        step = reel[int(idx)]
        cs = np.asarray(step["study"], dtype=np.float64)
        ce = np.asarray(step["exam"], dtype=np.float64)
        cy = np.asarray(step["y"], dtype=np.float64).reshape(-1)
        w = np.asarray(W[int(idx)], dtype=np.float64).reshape(3)
        picked_classrooms.append((cs, ce, cy))
        picked_weights.append(w)
        g = _ch6_avg_nll_grad_at_w(w, avg_roster)
        gn = float(np.linalg.norm(g))
        grad_dirs.append(g / gn if gn > 1e-9 else np.zeros(3, dtype=np.float64))
        grad_mags.append(gn)

    max_mag = max(grad_mags) if grad_mags else 1.0
    exag = 1.55
    grad_scales = [
        float(grad_3d_scale) * exag * (float(m) / max(max_mag, 1e-9))
        for m in grad_mags
    ]

    print(
        f"  avg-grad-variation: voxel caches (n_avg={len(avg_roster)}, "
        f"n_pick={n_pick}, in_zoom={len(in_bounds)})…",
        flush=True,
    )
    vox_avg_full = _ch6_cloud_workshop_voxel_cache_average(
        avg_roster, bounds=full_bounds, mu=mu, H=scene["H_avg"],
    )
    vox_avg_zoom = _ch6_cloud_workshop_voxel_cache_average(
        avg_roster, bounds=zoom_bounds, mu=mu, H=scene["H_avg"],
    )
    vox_by_bounds: dict = {
        tuple(float(v) for v in full_bounds): vox_avg_full,
        tuple(float(v) for v in zoom_bounds): vox_avg_zoom,
    }

    visit_classrooms = list(scene["classrooms"])
    visit_weights = list(scene["weights"])
    vox_each_visit: list = []
    for cs, ce, cy in visit_classrooms:
        vox_each_visit.append(
            _ch6_cloud_workshop_voxel_cache(cs, ce, cy, bounds=full_bounds),
        )

    return dict(
        scene=scene,
        pop_pack=pop_pack,
        avg_roster=avg_roster,
        pick_idx=pick_idx,
        picked_classrooms=picked_classrooms,
        picked_weights=picked_weights,
        grad_dirs=grad_dirs,
        grad_scales=grad_scales,
        full_bounds=full_bounds,
        zoom_bounds=zoom_bounds,
        grad_3d_scale=grad_3d_scale,
        grad_3d_axis_lim=grad_3d_axis_lim,
        vox_avg_full=vox_avg_full,
        vox_avg_zoom=vox_avg_zoom,
        vox_by_bounds=vox_by_bounds,
        visit_classrooms=visit_classrooms,
        visit_weights=visit_weights,
        vox_each_visit=vox_each_visit,
        n_demo=int(n_demo),
        draft=_CH3_DRAFT,
    )


def _ch6_classroom_size_workshop_scene(
    *,
    n_class: int,
    n_demo: int = 10,
    seed: int = 141,
    seed_from_key: str | None = None,
):
    """Workshop scene: parameter cloud from ``n_class`` classrooms; demo roster = first ``n_demo`` landings."""
    n_demo = int(n_demo)
    n_reel = max(n_demo + 8, int(CH6_N_POP_REEL))
    pack = ch6_population_param_cloud_pack(
        int(n_class),
        seed=int(seed),
        seed_from_key=seed_from_key,
        n_reel=n_reel,
    )
    W = np.asarray(pack["landed"], dtype=np.float64)
    mu = np.mean(W, axis=0)
    axis_lim = CH6_POPULATION_PARAM_AXIS_LIM
    ax_lo, ax_hi = float(axis_lim[0]), float(axis_lim[1])
    full_bounds = (ax_lo, ax_hi, ax_lo, ax_hi, ax_lo, ax_hi)
    reel = pack["reel_steps"]

    demo_idx = list(range(n_demo))
    classrooms: list = []
    weights: list = []
    for i in demo_idx:
        step = reel[i]
        classrooms.append((
            np.asarray(step["study"], dtype=np.float64),
            np.asarray(step["exam"], dtype=np.float64),
            np.asarray(step["y"], dtype=np.float64).reshape(-1),
        ))
        weights.append(np.asarray(W[i], dtype=np.float64).reshape(3))

    cs0, ce0, cy0 = classrooms[0]
    H_avg = ch6_observed_information(
        mu,
        ch6_design(cs0, ce0),
        np.asarray(cy0).reshape(-1),
        ridge=CH6_RIDGE,
    )
    newton_dir_mu = _ch6_newton_ascent_dir(mu, ch6_design(cs0, ce0), np.asarray(cy0).reshape(-1))
    ct_elev = ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV")))
    azim_end = float(_g("CH3_LIK_W12_CT_AZIM")) + 720.0
    w12_span = float(full_bounds[1]) - float(full_bounds[0])
    grad_3d_scale = 0.20 * w12_span
    grad_3d_axis_lim = (full_bounds[0], full_bounds[1])
    marker_s = ch6_population_n_sweep_marker_size(int(n_class))
    w0 = weights[0]

    info = dict(
        cs=cs0,
        ce=ce0,
        cy=cy0,
        w_knob=w0,
        xlim=pack["xlim"],
        ylim=pack["ylim"],
        marker_s=marker_s,
        belief_z_lim=axis_lim,
    )

    return dict(
        W=W,
        mu=mu,
        axis_lim=axis_lim,
        full_bounds=full_bounds,
        panel_xlim=pack["xlim"],
        panel_ylim=pack["ylim"],
        pop_s=pack["pop_s"],
        pop_e=pack["pop_e"],
        pop_y=pack["pop_y"],
        classrooms=tuple(classrooms),
        weights=tuple(weights),
        demo_idx=demo_idx,
        H_avg=H_avg,
        newton_dir_mu=newton_dir_mu,
        ct_elev=ct_elev,
        azim_end=azim_end,
        grad_3d_scale=grad_3d_scale,
        grad_3d_axis_lim=grad_3d_axis_lim,
        marker_s=marker_s,
        belief_z=axis_lim,
        info=info,
        n_class=int(n_class),
        w_class=w0,
        study=cs0,
        exam=ce0,
        y=cy0,
    )


def _ch6_demo_bright_mask(indices, n_pts):
    mask = np.zeros(int(n_pts), dtype=bool)
    for idx in indices:
        mask[int(idx)] = True
    return mask


def _ch6_black_points_kw(n_pts, active_indices, *, pulse_idx=None, pulse_u=None, pulse_gain=2.6):
    """Black markers for active classrooms; optional pulse on one index."""
    n = int(n_pts)
    scales = np.ones(n, dtype=np.float64)
    override = np.zeros((n, 4), dtype=np.float64)
    for idx in active_indices:
        override[int(idx)] = (0.0, 0.0, 0.0, 1.0)
    if pulse_idx is not None and pulse_u is not None:
        bump = np.sin(float(pulse_u) * np.pi)
        scales[int(pulse_idx)] = 1.0 + float(pulse_gain) * bump
    return dict(
        marker_s_per_point=scales,
        marker_color_override=override,
        marker_bright_mask=_ch6_demo_bright_mask(active_indices, n),
        marker_dim_scale=0.14,
        marker_below_voxels=True,
    )


def _ch6_voxel_sweep_layer(
    vox,
    bounds,
    *,
    sweep_u=1.0,
    prev=None,
    blend_u=None,
    wave_amp=0.0,
    reveal=False,
    uniform_alpha=None,
):
    if uniform_alpha is None:
        uniform_alpha = float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA)
    kw = dict(
        voxel_view_bounds=bounds,
        nll_voxel_cache=vox,
        nll_voxel_sweep_u=float(sweep_u),
        nll_voxel_alpha_u=1.0,
        nll_voxel_uniform_alpha=float(uniform_alpha),
        nll_voxel_sweep_wave_amp=float(wave_amp),
        nll_voxel_sweep_reveal_blend=bool(reveal),
    )
    if prev is not None:
        kw["nll_voxel_prev_cache"] = prev
        kw["nll_voxel_color_blend_u"] = float(blend_u if blend_u is not None else sweep_u)
    return kw


def _ch6_fig_to_pil(fig):
    arr = _g("fig_to_image")(fig, dpi=_g("CH3_ANIM_DPI"))
    plt.close(fig)
    from PIL import Image

    return Image.fromarray(np.asarray(arr, dtype=np.uint8))


def _ch6_render_classroom_grid_cell(
    scene,
    *,
    point_idx: int,
    classroom: tuple,
    bounds,
    voxel_cache,
    sweep_u: float = 1.0,
    wave_amp: float = 0.0,
):
    """One full ch4 duo frame for a classroom (same layout as ch5 2×2 grids)."""
    s = scene
    cs, ce, cy = classroom
    n_pts = len(s["W"])
    return _ch6_movement_plot_raw(
        scene,
        base_study=cs,
        base_exam=ce,
        base_y=cy,
        grey_u=0.90,
        population_fade_u=1.0,
        marker_alpha_scale=0.72,
        knob_w=s["W"][int(point_idx)],
        **_ch6_black_points_kw(n_pts, [int(point_idx)]),
        **_ch6_voxel_sweep_layer(
            voxel_cache, bounds, sweep_u=float(sweep_u), wave_amp=float(wave_amp),
        ),
    )


def _ch6_duo_grid_reference_cell(scene):
    """Single duo frame used only to size empty 2×2 quadrants."""
    return _ch6_movement_plot_raw(
        scene,
        grey_u=1.0,
        population_fade_u=0.0,
        marker_alpha_scale=0.0,
    )


def _ch6_composite_2x2_classroom_cells(cell_images, *, ref_cell=None):
    """2×2 grid of full duo frames (ch5 layout, gap=0)."""
    from PIL import Image

    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    grid = [[None, None], [None, None]]
    for k, (i, j) in enumerate(slots):
        if k < len(cell_images):
            grid[i][j] = cell_images[k]
    ref = ref_cell or next((c for c in cell_images if c is not None), None)
    if ref is None:
        raise ValueError("2×2 classroom grid needs at least one cell or ref_cell")
    cw, ch = ref.size
    out = Image.new("RGB", (2 * cw, 2 * ch), (255, 255, 255))
    grey = Image.new("RGB", (cw, ch), (242, 242, 242))
    for i in range(2):
        for j in range(2):
            im = grid[i][j]
            if im is None:
                im = grey
            elif im.size != (cw, ch):
                im = im.resize((cw, ch), resample=Image.Resampling.LANCZOS)
            out.paste(im, (j * cw, i * ch))
    return out


def _append_ch6_tutorial_morph(
    frames,
    clip_id,
    plot_img,
    *,
    scene,
    n_layout,
    n_panel,
    bottom_blocks,
    right_blocks,
    bottom_title,
    right_title,
    smooth,
    n_hold,
    right_title_single_line=True,
):
    """Shrink plot → reveal tutorial rails → write text."""
    plot_rect = _g("CH4_LIK_PLOT_START_RECT")
    for k in range(max(int(n_layout), 1)):
        u = smooth(float(k) / max(int(n_layout) - 1, 1))
        img = _ch6_compose_tutorial_frame(
            plot_img,
            layout_u=u,
            panel_u=0.0,
            title_u=0.0,
            write_u=0.0,
            bottom_blocks=bottom_blocks,
            right_blocks=right_blocks,
            bottom_prog=_ch6_block_write_prog_uniform(bottom_blocks, 0.0),
            right_prog=_ch6_block_write_prog_uniform(right_blocks, 0.0),
            bottom_title=bottom_title,
            right_title=right_title,
            plot_start_rect=plot_rect,
            right_title_single_line=right_title_single_line,
        )
        frames.append(img)
    for k in range(max(int(n_panel), 1)):
        u = smooth(float(k) / max(int(n_panel) - 1, 1))
        img = _ch6_compose_tutorial_frame(
            plot_img,
            layout_u=1.0,
            panel_u=u,
            title_u=u,
            write_u=u,
            bottom_blocks=bottom_blocks,
            right_blocks=right_blocks,
            bottom_prog=_ch6_block_write_prog_uniform(bottom_blocks, u),
            right_prog=_ch6_block_write_prog_uniform(right_blocks, u),
            bottom_title=bottom_title,
            right_title=right_title,
            plot_start_rect=plot_rect,
            right_title_single_line=right_title_single_line,
        )
        frames.append(img)
    frames.extend(_hold(frames[-1], max(1, n_hold // 2)))


def _append_ch6_tutorial_unmorph(
    frames,
    plot_img,
    *,
    n_layout,
    n_panel,
    bottom_blocks,
    right_blocks,
    bottom_title,
    right_title,
    smooth,
    right_title_single_line=True,
):
    """Fade tutorial rails out → full-bleed plot."""
    plot_rect = _g("CH4_LIK_PLOT_START_RECT")
    for k in range(max(int(n_panel), 1)):
        u = 1.0 - smooth(float(k) / max(int(n_panel) - 1, 1))
        frames.append(_ch6_compose_tutorial_frame(
            plot_img,
            layout_u=1.0,
            panel_u=u,
            title_u=u,
            write_u=u,
            bottom_blocks=bottom_blocks,
            right_blocks=right_blocks,
            bottom_prog=_ch6_block_write_prog_uniform(bottom_blocks, u),
            right_prog=_ch6_block_write_prog_uniform(right_blocks, u),
            bottom_title=bottom_title,
            right_title=right_title,
            plot_start_rect=plot_rect,
            right_title_single_line=right_title_single_line,
        ))
    for k in range(max(int(n_layout), 1)):
        u = 1.0 - smooth(float(k) / max(int(n_layout) - 1, 1))
        frames.append(_ch6_compose_tutorial_frame(
            plot_img,
            layout_u=u,
            panel_u=0.0,
            title_u=0.0,
            write_u=0.0,
            bottom_blocks=bottom_blocks,
            right_blocks=right_blocks,
            bottom_prog=_ch6_block_write_prog_uniform(bottom_blocks, 0.0),
            right_prog=_ch6_block_write_prog_uniform(right_blocks, 0.0),
            bottom_title=bottom_title,
            right_title=right_title,
            plot_start_rect=plot_rect,
            right_title_single_line=right_title_single_line,
        ))


def _build_ch6_141_movement_formula_layout_from_133(clip_id):
    """ch6_133 pre-push hold → tutorial rails; derive movement ≈ grad / curvature."""
    scene = _ch6_movement_story_pack()
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    frames: list = []

    pre_kw = _ch6_movement_pre_move_kw(scene)
    plot_pre = _ch6_movement_plot_raw(scene, **pre_kw)
    frames.extend(_hold(_finish(plot_pre, clip_id), n_hold))

    bottom = _ch6_movement_bottom_blocks(0)
    right = _ch6_movement_right_blocks(0)
    _append_ch6_tutorial_morph(
        frames, clip_id, plot_pre,
        scene=scene,
        n_layout=_draft_short(28, 12),
        n_panel=_draft_short(24, 10),
        bottom_blocks=bottom,
        right_blocks=right,
        bottom_title=CH6_MOVEMENT_BOTTOM_TITLE,
        right_title=CH6_MOVEMENT_RIGHT_TITLE,
        smooth=smooth,
        n_hold=n_hold,
    )

    for stage in range(1, 4):
        bottom = _ch6_movement_bottom_blocks(stage)
        right = _ch6_movement_right_blocks(stage)
        shell_key = _ch6_movement_shell_key(right)
        n_x = _draft_short(18, 8)
        for k in range(n_x):
            u = smooth(float(k) / max(n_x - 1, 1))
            frames.append(_ch6_compose_tutorial_frame(
                plot_pre,
                layout_u=1.0,
                panel_u=1.0,
                title_u=1.0,
                write_u=1.0,
                bottom_blocks=bottom,
                right_blocks=right,
                bottom_prog=_ch6_block_write_prog_uniform(bottom, u),
                right_prog=_ch6_block_write_prog_uniform(right, u),
                bottom_title=CH6_MOVEMENT_BOTTOM_TITLE,
                right_title=CH6_MOVEMENT_RIGHT_TITLE,
                right_title_single_line=True,
                shell_cache_key=shell_key,
            ))
        frames.extend(_hold(frames[-1], max(1, n_hold // 2)))

    for rs in range(2, 5):
        bottom = _ch6_movement_bottom_blocks(3)
        right = _ch6_movement_right_blocks(rs)
        shell_key = _ch6_movement_shell_key(right)
        n_x = _draft_short(16, 7)
        for k in range(n_x):
            u = smooth(float(k) / max(n_x - 1, 1))
            frames.append(_ch6_compose_tutorial_frame(
                plot_pre,
                layout_u=1.0,
                panel_u=1.0,
                title_u=1.0,
                write_u=1.0,
                bottom_blocks=bottom,
                right_blocks=right,
                bottom_prog=_ch6_block_write_prog_uniform(bottom, 1.0),
                right_prog=_ch6_block_write_prog_uniform(right, u),
                bottom_title=CH6_MOVEMENT_BOTTOM_TITLE,
                right_title=CH6_MOVEMENT_RIGHT_TITLE,
                right_title_single_line=True,
                shell_cache_key=shell_key,
            ))
        frames.extend(_hold(frames[-1], max(1, n_hold // 3)))

    frames.extend(_hold(frames[-1], n_hold * 2))
    return frames


def _build_ch6_142_classroom_voxel_average_reverse_push_from_141(clip_id):
    """Tutorial out → cloud → incremental voxel average → reverse push (black arrow)."""
    scene = _ch6_movement_story_pack()
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    s = scene
    frames: list = []

    pre_kw = _ch6_movement_pre_move_kw(scene)
    plot_pre = _ch6_movement_plot_raw(scene, **pre_kw)
    bottom_end = _ch6_movement_bottom_blocks(3)
    right_end = _ch6_movement_right_blocks(4)
    _append_ch6_tutorial_unmorph(
        frames, plot_pre,
        n_layout=_draft_short(24, 10),
        n_panel=_draft_short(20, 8),
        bottom_blocks=bottom_end,
        right_blocks=right_end,
        bottom_title=CH6_MOVEMENT_BOTTOM_TITLE,
        right_title=CH6_MOVEMENT_RIGHT_TITLE,
        smooth=smooth,
    )

    zoom_bounds = s["zoom_bounds"]
    ax_lo, ax_hi = float(s["axis_lim"][0]), float(s["axis_lim"][1])
    # Same full cloud / axis window as ch6_129–131.
    full_bounds = (ax_lo, ax_hi, ax_lo, ax_hi, ax_lo, ax_hi)

    def _lerp_bounds(u, start_bounds, end_bounds):
        u = float(u)
        return tuple(
            float(a) + u * (float(b) - float(a))
            for a, b in zip(start_bounds, end_bounds)
        )

    def _callouts(avg_u=1.0, class_u=1.0):
        out = []
        if float(avg_u) > 1e-4:
            out.append(dict(
                point=s["mu"], label="average",
                color=CH6_VARIANCE_RED, u=float(avg_u),
                label_placement="bottom_outside",
            ))
        if float(class_u) > 1e-4:
            out.append(dict(
                point=s["w_class"], label="this classroom",
                color="#111111", u=float(class_u),
                label_placement="top_outside",
            ))
        return out or None

    def _points_kw(**extra):
        """Highlighted average + classroom (population_fade=1 keeps markers visible)."""
        kw = dict(
            population_fade_u=1.0,
            marker_alpha_scale=0.0,
            highlight_mu_u=1.0,
            highlight_classroom_u=1.0,
            highlight_classroom_color="#111111",
            highlight_classroom_w=s["w_class"],
            highlight_mu_w=s["mu"],
            ch6_pair_line_u=1.0,
            ch6_pair_line_alpha=1.0,
            ch6_callouts=_callouts(),
            nll_voxel_cache=None,
            nll_voxel_sweep_u=0.0,
            grey_u=0.0,
            knob_w=s["w_class"],
        )
        kw.update(extra)
        return kw

    def _emit(**kw):
        return _finish(_ch6_movement_plot_raw(scene, **kw), clip_id)

    def _vox_layer(cache, bounds):
        return dict(
            voxel_view_bounds=bounds,
            nll_voxel_cache=cache,
            nll_voxel_sweep_u=1.0,
            nll_voxel_alpha_u=1.0,
            nll_voxel_uniform_alpha=float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA),
        )

    def _avg_vox(roster, bounds):
        """Fixed cells-per-axis for *bounds* — same apparent density as ch6_133 at any zoom."""
        return _ch6_nll_voxel_cache_average(
            roster, bounds=bounds, mu=s["mu"], H=s["H_avg"], cache_kw={},
        )

    def _demo_bright_mask(indices):
        mask = np.zeros(len(s["W"]), dtype=bool)
        for idx in indices:
            mask[int(idx)] = True
        return mask

    def _cloud_kw(*, active_demo=None, grey_u=0.62, vox=None, bounds=None, **extra):
        kw = dict(
            population_fade_u=1.0,
            marker_alpha_scale=0.78,
            grey_u=float(grey_u),
            highlight_mu_u=0.65,
            highlight_mu_w=s["mu"],
            highlight_classroom_u=0.45,
            highlight_classroom_w=s["w_class"],
            ch6_pair_line_u=0.25,
            knob_w=s["mu"],
        )
        if active_demo is not None:
            kw.update(
                marker_bright_mask=_demo_bright_mask(active_demo),
                marker_dim_scale=0.14,
            )
        if bounds is not None:
            if vox is not None:
                kw.update(**_vox_layer(vox, bounds))
            else:
                kw["voxel_view_bounds"] = bounds
        kw.update(extra)
        return kw

    demo_idx = _ch6_pick_classroom_demo_indices(s["W"], s["w_class"], n=5, seed=141)
    classrooms = _ch6_classroom_roster_for_indices(scene, demo_idx)

    n_vox_out = _draft_short(20, 8)
    for k in range(n_vox_out):
        u = 1.0 - smooth(float(k) / max(n_vox_out - 1, 1))
        frames.append(_emit(**_points_kw(
            voxel_view_bounds=zoom_bounds,
            nll_voxel_cache=s["voxel_class"],
            nll_voxel_sweep_u=1.0,
            nll_voxel_alpha_u=u,
            nll_voxel_uniform_alpha=float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA),
        )))
    frames.extend(_hold(_emit(**_points_kw(voxel_view_bounds=zoom_bounds)), max(1, n_hold // 3)))

    # Zoom out axis limits: ch6_133 window → ch6_131 cloud window (two points stay).
    n_zoom_out = _draft_short(40, 16)
    for k in range(n_zoom_out):
        u = smooth(float(k) / max(n_zoom_out - 1, 1))
        frames.append(_emit(**_points_kw(
            voxel_view_bounds=_lerp_bounds(u, zoom_bounds, full_bounds),
        )))
    frames.extend(_hold(frames[-1], max(1, n_hold // 4)))

    # Full cloud at cloud scale — grey first, then emphasize averaging classrooms.
    n_cloud = _draft_short(24, 10)
    for k in range(n_cloud):
        u = smooth(float(k) / max(n_cloud - 1, 1))
        frames.append(_emit(**_cloud_kw(
            bounds=full_bounds,
            grey_u=0.88 * u,
            marker_alpha_scale=0.78 * u,
            highlight_mu_u=0.55 + 0.10 * u,
            highlight_classroom_u=0.55 + 0.10 * u,
            ch6_pair_line_u=0.35 + 0.65 * u,
        )))
    frames.extend(_hold(_emit(**_cloud_kw(bounds=full_bounds, grey_u=0.88)), max(1, n_hold // 5)))

    n_emph = _draft_short(28, 10)
    for k in range(n_emph):
        u = smooth(float(k) / max(n_emph - 1, 1))
        frames.append(_emit(**_cloud_kw(
            bounds=full_bounds,
            grey_u=0.88,
            marker_bright_mask=_demo_bright_mask(demo_idx),
            marker_dim_scale=0.14 + 0.72 * (1.0 - u),
        )))
    frames.extend(_hold(_emit(**_cloud_kw(
        bounds=full_bounds, grey_u=0.88, active_demo=demo_idx,
    )), max(1, n_hold // 5)))

    accum: list = []
    vox_full = None
    vox_prev = None
    for ki, roster in enumerate(classrooms):
        accum.append(roster)
        active = demo_idx[: ki + 1]
        vox_avg = _avg_vox(accum, full_bounds)
        if vox_prev is None:
            n_sweep = _draft_short(40, 16)
            for k in range(n_sweep):
                u = smooth(float(k) / max(n_sweep - 1, 1))
                frames.append(_emit(**_cloud_kw(
                    bounds=full_bounds,
                    grey_u=0.88,
                    active_demo=active,
                    vox=vox_avg,
                    nll_voxel_sweep_u=u,
                    nll_voxel_alpha_u=u,
                )))
        else:
            n_blend = _draft_short(28, 12)
            for k in range(n_blend):
                u = smooth(float(k) / max(n_blend - 1, 1))
                frames.append(_emit(**_cloud_kw(
                    bounds=full_bounds,
                    grey_u=0.88,
                    active_demo=active,
                    vox=vox_avg,
                    nll_voxel_prev_cache=vox_prev,
                    nll_voxel_color_blend_u=u,
                    nll_voxel_sweep_u=1.0,
                    nll_voxel_alpha_u=1.0,
                )))
        vox_prev = vox_avg
        vox_full = vox_avg
        frames.extend(_hold(frames[-1], max(1, n_hold // 5)))

    big_roster = _ch6_movement_refined_roster(scene)
    vox_refined = _avg_vox(big_roster, full_bounds)
    n_refine = _draft_short(32, 12)
    for k in range(n_refine):
        u = smooth(float(k) / max(n_refine - 1, 1))
        frames.append(_emit(**_cloud_kw(
            bounds=full_bounds,
            grey_u=0.88,
            active_demo=demo_idx,
            vox=vox_refined,
            nll_voxel_prev_cache=vox_full,
            nll_voxel_color_blend_u=u,
            nll_voxel_sweep_u=1.0,
            nll_voxel_alpha_u=1.0,
        )))
    vox_full = vox_refined
    frames.extend(_hold(frames[-1], n_hold))

    vox_zoom = _ch6_movement_zoom_voxel_cache(scene, roster=big_roster)
    vox_by_bounds: dict = {
        tuple(float(v) for v in zoom_bounds): vox_zoom,
        tuple(float(v) for v in full_bounds): vox_full,
    }

    def _vox_for_bounds(bounds):
        key = tuple(float(v) for v in bounds)
        hit = vox_by_bounds.get(key)
        if hit is None:
            hit = _avg_vox(big_roster, bounds)
            vox_by_bounds[key] = hit
        return hit

    def _zoom_iso_kw(**extra):
        base = dict(
            population_fade_u=1.0,
            marker_alpha_scale=0.0,
            marker_bright_mask=_demo_bright_mask([demo_idx[0]]),
            marker_dim_scale=0.10,
            grey_u=0.92,
            highlight_mu_u=1.0,
            highlight_mu_w=s["mu"],
            highlight_classroom_u=1.0,
            highlight_classroom_color="#111111",
            highlight_classroom_w=s["w_class"],
            ch6_pair_line_u=1.0,
            ch6_pair_line_alpha=1.0,
            knob_w=s["w_class"],
            **_vox_layer(vox_zoom, zoom_bounds),
        )
        base.update(extra)
        return base

    # Zoom in: rebuild voxel grid at each window so cells-per-axis stays constant (ch6_133).
    n_zoom_in = _draft_short(40, 16)
    for k in range(n_zoom_in):
        u = smooth(float(k) / max(n_zoom_in - 1, 1))
        b6 = _lerp_bounds(u, full_bounds, zoom_bounds)
        frames.append(_emit(
            population_fade_u=1.0,
            marker_alpha_scale=0.78 * (1.0 - u),
            grey_u=0.88 + 0.04 * u,
            marker_bright_mask=_demo_bright_mask(demo_idx),
            marker_dim_scale=0.14,
            **_vox_layer(_vox_for_bounds(b6), b6),
            highlight_mu_u=0.65 + 0.35 * u,
            highlight_mu_w=s["mu"],
            highlight_classroom_u=0.65 + 0.35 * u,
            highlight_classroom_w=s["w_class"],
            highlight_classroom_color="#111111",
            ch6_pair_line_u=0.35 + 0.65 * u,
            ch6_callouts=_callouts(avg_u=0.65 + 0.35 * u, class_u=0.65 + 0.35 * u),
            knob_w=s["w_class"],
        ))
    frames.extend(_hold(_emit(**_zoom_iso_kw()), max(1, n_hold // 3)))

    # Pre-spring setup (mirror ch6_133): 3-D gradient → 2-D thresholds.
    n_grad3d = _draft_short(28, 10)
    for k in range(n_grad3d):
        u = smooth(float(k) / max(n_grad3d - 1, 1))
        frames.append(_emit(**_zoom_iso_kw(
            nll_grad_3d=s["newton_dir_mu"],
            nll_grad_3d_w=s["w_class"],
            nll_grad_3d_u=u,
            nll_grad_3d_is_direction=True,
            nll_grad_3d_color="#111111",
        )))
    frames.extend(_hold(frames[-1], max(1, n_hold // 4)))

    n_dist_out = _draft_short(20, 8)
    for k in range(n_dist_out):
        u = smooth(float(k) / max(n_dist_out - 1, 1))
        frames.append(_emit(**_zoom_iso_kw(
            mu_threshold_2d=s["mu"],
            mu_threshold_2d_u=u,
            ghost_threshold_2d=s["mu"],
            ghost_threshold_2d_u=u,
            classroom_threshold_2d=s["w_class"],
            classroom_threshold_2d_u=u,
            show_2d_grads=u > 0.35,
            grad_w_live=s["w_class"],
            nll_grad_3d=s["newton_dir_mu"],
            nll_grad_3d_w=s["w_class"],
            nll_grad_3d_u=1.0,
            nll_grad_3d_is_direction=True,
            nll_grad_3d_color="#111111",
            ch6_callouts=_callouts(avg_u=1.0, class_u=1.0),
        )))
    frames.extend(_hold(_emit(**_zoom_iso_kw(
        mu_threshold_2d=s["mu"],
        mu_threshold_2d_u=1.0,
        ghost_threshold_2d=s["mu"],
        ghost_threshold_2d_u=1.0,
        classroom_threshold_2d=s["w_class"],
        classroom_threshold_2d_u=1.0,
        show_2d_grads=True,
        grad_w_live=s["w_class"],
        nll_grad_3d=s["newton_dir_mu"],
        nll_grad_3d_w=s["w_class"],
        nll_grad_3d_u=1.0,
        nll_grad_3d_is_direction=True,
        nll_grad_3d_color="#111111",
        ch6_callouts=_callouts(avg_u=1.0, class_u=1.0),
    )), max(1, n_hold // 3)))

    # Reverse spring: classroom → average (ch6_133 spring mirrored).
    H = s["H_avg"]
    w_target = s["mu"]
    w_start = s["w_class"]
    spring_path = _hessian_restore_path(
        w_start[:2], w_target[:2], H[:2, :2],
        n_frames=_draft_short(44, 18),
        damp=1.15,
        stiffness=5.0,
    )
    newton_path_3d = _ch6_damped_newton_path_3d(
        w_start, w_target, s["Xd"], s["y"],
        n_frames=len(spring_path),
        axis_lim=s["axis_lim"],
        pull_gain=0.55,
    )
    newton_path_3d[-1] = w_target.copy()
    for i in range(1, len(spring_path)):
        t = float(i) / max(len(spring_path) - 1, 1)
        xy = spring_path[i]
        w_walk = np.asarray(newton_path_3d[i], dtype=np.float64).reshape(3).copy()
        if i == len(spring_path) - 1:
            w_walk = w_target.copy()
        ww_line = w_start.copy()
        ww_line[0], ww_line[1] = float(xy[0]), float(xy[1])
        ww_line[2] = (1.0 - t) * float(w_start[2]) + t * float(w_target[2])
        if i == len(spring_path) - 1:
            ww_line = w_target.copy()
        newton_dir = _ch6_newton_ascent_dir(w_walk, s["Xd"], s["y"])
        opaque_u = float(CH6_NLL_VOXEL_WALKER_OPAQUE_MAX) * smooth(0.2 + 0.8 * t)
        frames.append(_emit(**_zoom_iso_kw(
            highlight_classroom_w=w_walk,
            highlight_mu_w=s["mu"],
            mu_threshold_2d=s["mu"],
            mu_threshold_2d_u=1.0,
            ghost_threshold_2d=s["mu"],
            ghost_threshold_2d_u=1.0,
            classroom_threshold_2d=ww_line,
            classroom_threshold_2d_u=1.0,
            knob_w=ww_line,
            show_2d_grads=True,
            grad_w_live=ww_line,
            nll_grad_3d=newton_dir,
            nll_grad_3d_w=w_walk,
            nll_grad_3d_u=max(0.12, 1.0 - t * 0.82),
            nll_grad_3d_is_direction=True,
            nll_grad_3d_color="#111111",
            nll_voxel_walker_w=w_walk,
            nll_voxel_walker_opaque_u=opaque_u,
            ch6_callouts=_callouts(avg_u=1.0, class_u=1.0),
        )))

    frames.extend(_hold(frames[-1], n_hold * 2))
    return frames


def _build_ch6_144_voxel_average_wave_workshop(clip_id):
    """Workshop extract: ch6_142 cloud voxel-averaging only, with diagonal wave sweeps."""
    from ch6_frequentist import _CH3_DRAFT

    scene = _ch6_movement_story_pack()
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    s = scene
    frames: list = []
    ax_lo, ax_hi = float(s["axis_lim"][0]), float(s["axis_lim"][1])
    full_bounds = (ax_lo, ax_hi, ax_lo, ax_hi, ax_lo, ax_hi)
    wave_amp = _ch6_nll_voxel_sweep_wave_amp()
    vox_uniform = float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA)

    def _emit(**kw):
        return _finish(_ch6_movement_plot_raw(scene, **kw), clip_id)

    def _avg_vox(roster, bounds):
        return _ch6_nll_voxel_cache_average(
            roster, bounds=bounds, mu=s["mu"], H=s["H_avg"], cache_kw={},
        )

    def _demo_bright_mask(indices):
        mask = np.zeros(len(s["W"]), dtype=bool)
        for idx in indices:
            mask[int(idx)] = True
        return mask

    def _cloud_base(**extra):
        kw = dict(
            population_fade_u=1.0,
            marker_alpha_scale=0.78,
            grey_u=0.88,
            highlight_mu_u=0.65,
            highlight_mu_w=s["mu"],
            highlight_classroom_u=0.45,
            highlight_classroom_w=s["w_class"],
            ch6_pair_line_u=0.25,
            knob_w=s["mu"],
            voxel_view_bounds=full_bounds,
        )
        kw.update(extra)
        return kw

    def _new_point_kw(new_idx, *, pulse_u=None):
        """Black marker; optional sin pulse on the classroom just added to the average."""
        n = len(s["W"])
        scales = np.ones(n, dtype=np.float64)
        override = np.zeros((n, 4), dtype=np.float64)
        idx = int(new_idx)
        override[idx] = (0.0, 0.0, 0.0, 1.0)
        if pulse_u is not None:
            bump = np.sin(float(pulse_u) * np.pi)
            scales[idx] = 1.0 + 2.6 * bump
        return dict(
            marker_s_per_point=scales,
            marker_color_override=override,
        )

    def _sweep_kw(vox, prev, u, active_demo, new_idx=None):
        kw = _cloud_base(
            marker_bright_mask=_demo_bright_mask(active_demo),
            marker_dim_scale=0.14,
            nll_voxel_cache=vox,
            nll_voxel_prev_cache=prev,
            nll_voxel_sweep_u=float(u),
            nll_voxel_alpha_u=1.0,
            nll_voxel_uniform_alpha=vox_uniform,
            nll_voxel_color_blend_u=float(u) if prev is not None else 1.0,
            nll_voxel_sweep_wave_amp=wave_amp,
            nll_voxel_sweep_reveal_blend=prev is not None,
        )
        if new_idx is not None:
            kw.update(_new_point_kw(new_idx, pulse_u=None))
        return kw

    def _append_new_point_pulse(frames, active, new_idx):
        n_pulse = _draft_short(28, 12)
        for k in range(n_pulse):
            pu = smooth(float(k) / max(n_pulse - 1, 1))
            frames.append(_emit(**_cloud_base(
                marker_bright_mask=_demo_bright_mask(active),
                marker_dim_scale=0.14,
                **_new_point_kw(new_idx, pulse_u=pu),
            )))
        frames.append(_emit(**_cloud_base(
            marker_bright_mask=_demo_bright_mask(active),
            marker_dim_scale=0.14,
            **_new_point_kw(new_idx, pulse_u=None),
        )))

    demo_idx = _ch6_pick_classroom_demo_indices(s["W"], s["w_class"], n=5, seed=141)
    classrooms = _ch6_classroom_roster_for_indices(scene, demo_idx)

    n_cloud = _draft_short(20, 8)
    for k in range(n_cloud):
        u = smooth(float(k) / max(n_cloud - 1, 1))
        frames.append(_emit(**_cloud_base(
            grey_u=0.88 * u,
            marker_alpha_scale=0.78 * u,
            highlight_mu_u=0.55 + 0.10 * u,
            highlight_classroom_u=0.55 + 0.10 * u,
            ch6_pair_line_u=0.35 + 0.65 * u,
        )))
    frames.extend(_hold(_emit(**_cloud_base()), max(1, n_hold // 5)))

    n_emph = _draft_short(24, 10)
    for k in range(n_emph):
        u = smooth(float(k) / max(n_emph - 1, 1))
        frames.append(_emit(**_cloud_base(
            marker_bright_mask=_demo_bright_mask(demo_idx),
            marker_dim_scale=0.14 + 0.72 * (1.0 - u),
        )))
    frames.extend(_hold(_emit(**_cloud_base(
        marker_bright_mask=_demo_bright_mask(demo_idx),
        marker_dim_scale=0.14,
    )), max(1, n_hold // 4)))

    accum: list = []
    vox_prev = None
    n_step_sweep = _draft_short(56, 22)
    for ki, roster in enumerate(classrooms):
        accum.append(roster)
        active = demo_idx[: ki + 1]
        new_idx = int(active[-1])
        _append_new_point_pulse(frames, active, new_idx)
        vox_avg = _avg_vox(accum, full_bounds)
        for k in range(n_step_sweep):
            u = smooth(float(k) / max(n_step_sweep - 1, 1))
            frames.append(_emit(**_sweep_kw(vox_avg, vox_prev, u, active, new_idx=new_idx)))
        vox_prev = vox_avg
        frames.extend(_hold(frames[-1], max(1, n_hold // 4)))

    n_avg = 20 if _CH3_DRAFT else 60
    rng = np.random.default_rng(144)
    big_roster = list(classrooms)
    while len(big_roster) < n_avg:
        cs, ce, cy, _ = ch6_draw_classroom_from_population(
            s["pop_s"], s["pop_e"], s["pop_y"], 100, rng=rng,
        )
        big_roster.append((cs, ce, cy))
    vox_refined = _avg_vox(big_roster, full_bounds)
    n_refine_sweep = _draft_short(64, 24)
    for k in range(n_refine_sweep):
        u = smooth(float(k) / max(n_refine_sweep - 1, 1))
        frames.append(_emit(**_sweep_kw(vox_refined, vox_prev, u, demo_idx)))
    frames.extend(_hold(frames[-1], n_hold * 2))
    return frames


def _build_ch6_voxel_average_classrooms_workshop(
    clip_id,
    *,
    n_demo: int = 10,
    n_class: int = 100,
    seed: int = 141,
    seed_from_key: str | None = None,
    adapt_2d_classroom: bool = False,
):
    """Incremental voxel average with black-point beats + wave sweeps."""
    scene = _ch6_classroom_size_workshop_scene(
        n_class=int(n_class),
        n_demo=int(n_demo),
        seed=int(seed),
        seed_from_key=seed_from_key,
    )
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD
    s = scene
    frames: list = []
    n_pts = len(s["W"])
    full_bounds = s["full_bounds"]
    wave_amp = _ch6_nll_voxel_sweep_wave_amp()
    vox_uniform = float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA)
    classrooms = list(s["classrooms"])
    demo_idx = s["demo_idx"]
    print(
        f"  {clip_id}: n_class={n_class} parameter cloud ({n_pts} landings)…",
        flush=True,
    )

    def _panel_for(ri: int):
        if not adapt_2d_classroom:
            return {}
        cs, ce, cy = classrooms[int(ri)]
        return _ch6_classroom_panel_kw(cs, ce, cy)

    def _emit(**kw):
        return _finish(_ch6_movement_plot_raw(scene, **kw), clip_id)

    def _avg_vox(roster):
        return _ch6_cloud_workshop_voxel_cache_average(
            roster, bounds=full_bounds, mu=s["mu"], H=s["H_avg"],
        )

    def _cloud_base(**extra):
        kw = dict(
            population_fade_u=1.0,
            marker_alpha_scale=0.78,
            grey_u=0.88,
            highlight_mu_u=0.65,
            highlight_mu_w=s["mu"],
            highlight_classroom_u=0.45,
            ch6_pair_line_u=0.25,
            knob_w=s["mu"],
            marker_below_voxels=True,
        )
        kw.update(extra)
        return kw

    def _append_point_pulse(new_idx, vox_prev, roster_i):
        n_pulse = _workshop_short(28, 12)
        for k in range(n_pulse):
            pu = smooth(float(k) / max(n_pulse - 1, 1))
            kw = _cloud_base(**_ch6_black_points_kw(
                n_pts, [int(new_idx)], pulse_idx=int(new_idx), pulse_u=pu,
            ), **_panel_for(roster_i))
            if vox_prev is not None:
                kw.update(_ch6_voxel_sweep_layer(vox_prev, full_bounds, sweep_u=1.0))
            frames.append(_emit(**kw))
        kw = _cloud_base(
            **_ch6_black_points_kw(n_pts, [int(new_idx)]),
            **_panel_for(roster_i),
        )
        if vox_prev is not None:
            kw.update(_ch6_voxel_sweep_layer(vox_prev, full_bounds, sweep_u=1.0))
        frames.append(_emit(**kw))

    panel0 = _panel_for(0)

    n_cloud = _workshop_short(20, 8)
    for k in range(n_cloud):
        u = smooth(float(k) / max(n_cloud - 1, 1))
        frames.append(_emit(**_cloud_base(
            grey_u=0.88 * u,
            marker_alpha_scale=0.78 * u,
            highlight_mu_u=0.55 + 0.10 * u,
            highlight_classroom_u=0.55 + 0.10 * u,
            ch6_pair_line_u=0.35 + 0.65 * u,
            **panel0,
        )))
    frames.extend(_hold(_emit(**_cloud_base(**panel0)), _workshop_hold(n_hold)))

    n_emph = _workshop_short(24, 10)
    for k in range(n_emph):
        u = smooth(float(k) / max(n_emph - 1, 1))
        frames.append(_emit(**_cloud_base(
            marker_bright_mask=_ch6_demo_bright_mask(demo_idx, n_pts),
            marker_dim_scale=0.14 + 0.72 * (1.0 - u),
            **panel0,
        )))
    frames.extend(_hold(_emit(**_cloud_base(
        marker_bright_mask=_ch6_demo_bright_mask(demo_idx, n_pts),
        marker_dim_scale=0.14,
        **panel0,
    )), _workshop_hold(n_hold)))

    accum: list = []
    vox_prev = None
    n_step_sweep = _workshop_short(56, 22)
    for ki, roster in enumerate(classrooms):
        accum.append(roster)
        new_idx = int(demo_idx[ki])
        _append_point_pulse(new_idx, vox_prev, ki)
        print(f"  {clip_id}: averaging voxel cache {ki + 1}/{len(classrooms)}…", flush=True)
        vox_avg = _avg_vox(accum)
        for k in range(n_step_sweep):
            u = smooth(float(k) / max(n_step_sweep - 1, 1))
            kw = _cloud_base(
                **_ch6_black_points_kw(n_pts, [new_idx]),
                **_panel_for(ki),
            )
            kw.update(_ch6_voxel_sweep_layer(
                vox_avg, full_bounds,
                sweep_u=u,
                prev=vox_prev,
                blend_u=u,
                wave_amp=wave_amp,
                reveal=vox_prev is not None,
                uniform_alpha=vox_uniform,
            ))
            frames.append(_emit(**kw))
        vox_prev = vox_avg
        frames.extend(_hold(frames[-1], _workshop_hold(n_hold)))

    frames.extend(_hold(frames[-1], _workshop_hold(n_hold * 2)))
    return frames


def _build_ch6_voxel_visit_classrooms_workshop(
    clip_id,
    *,
    n_demo: int = 10,
    n_class: int = 100,
    seed: int = 141,
    seed_from_key: str | None = None,
):
    """Visit each classroom's voxel landscape in turn; average revealed at end."""
    scene = _ch6_classroom_size_workshop_scene(
        n_class=int(n_class),
        n_demo=int(n_demo),
        seed=int(seed),
        seed_from_key=seed_from_key,
    )
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD
    s = scene
    frames: list = []
    n_pts = len(s["W"])
    full_bounds = s["full_bounds"]
    wave_amp = _ch6_nll_voxel_sweep_wave_amp()
    vox_uniform = float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA)
    classrooms = list(s["classrooms"])
    weights = list(s["weights"])
    demo_idx = s["demo_idx"]
    print(
        f"  {clip_id}: n_class={n_class} parameter cloud ({n_pts} landings)…",
        flush=True,
    )
    vox_each: list = []
    for i, (cs, ce, cy) in enumerate(classrooms):
        print(
            f"  {clip_id}: classroom voxel cache {i + 1}/{len(classrooms)}…",
            flush=True,
        )
        vox_each.append(_ch6_cloud_workshop_voxel_cache(cs, ce, cy, bounds=full_bounds))
    print(f"  {clip_id}: average voxel cache…", flush=True)
    vox_avg = _ch6_cloud_workshop_voxel_cache_average(
        classrooms, bounds=full_bounds, mu=s["mu"], H=s["H_avg"],
    )

    def _emit(**kw):
        return _finish(_ch6_movement_plot_raw(scene, **kw), clip_id)

    def _cloud_base(**extra):
        kw = dict(
            population_fade_u=1.0,
            marker_alpha_scale=0.78,
            grey_u=0.88,
            highlight_mu_u=0.65,
            highlight_mu_w=s["mu"],
            highlight_classroom_u=0.45,
            ch6_pair_line_u=0.25,
            knob_w=s["mu"],
            marker_below_voxels=True,
        )
        kw.update(extra)
        return kw

    def _append_classroom_pulse(panel_kw, vox_show, vox_bg, cloud_idx):
        n_pulse = _workshop_short(28, 12)
        for k in range(n_pulse):
            pu = smooth(float(k) / max(n_pulse - 1, 1))
            bump = np.sin(float(pu) * np.pi)
            kw = _cloud_base(
                **_ch6_black_points_kw(
                    n_pts, [int(cloud_idx)], pulse_idx=int(cloud_idx), pulse_u=pu,
                ),
                **panel_kw,
                highlight_classroom_u=0.35 + 0.65 * bump,
            )
            if vox_bg is not None:
                kw.update(_ch6_voxel_sweep_layer(vox_bg, full_bounds, sweep_u=1.0))
            else:
                kw.update(_ch6_voxel_sweep_layer(
                    vox_show, full_bounds, sweep_u=0.0,
                ))
            frames.append(_emit(**kw))
        kw = _cloud_base(
            **_ch6_black_points_kw(n_pts, [int(cloud_idx)]),
            **panel_kw,
        )
        if vox_bg is not None:
            kw.update(_ch6_voxel_sweep_layer(vox_bg, full_bounds, sweep_u=1.0))
        frames.append(_emit(**kw))

    panel0 = _ch6_visit_classroom_panel_kw(0, classrooms, weights)

    n_cloud = _workshop_short(20, 8)
    for k in range(n_cloud):
        u = smooth(float(k) / max(n_cloud - 1, 1))
        frames.append(_emit(**_cloud_base(
            grey_u=0.88 * u,
            marker_alpha_scale=0.78 * u,
            highlight_mu_u=0.55 + 0.10 * u,
            highlight_classroom_u=0.55 + 0.10 * u,
            ch6_pair_line_u=0.35 + 0.65 * u,
            **panel0,
        )))
    frames.extend(_hold(_emit(**_cloud_base(**panel0)), _workshop_hold(n_hold)))

    n_emph = _workshop_short(24, 10)
    for k in range(n_emph):
        u = smooth(float(k) / max(n_emph - 1, 1))
        frames.append(_emit(**_cloud_base(
            marker_bright_mask=_ch6_demo_bright_mask(demo_idx, n_pts),
            marker_dim_scale=0.14 + 0.72 * (1.0 - u),
            **panel0,
        )))
    frames.extend(_hold(_emit(**_cloud_base(
        marker_bright_mask=_ch6_demo_bright_mask(demo_idx, n_pts),
        marker_dim_scale=0.14,
        **panel0,
    )), _workshop_hold(n_hold)))

    n_step_sweep = _workshop_short(56, 22)
    vox_prev = None
    for ki, vox_curr in enumerate(vox_each):
        panel_kw = _ch6_visit_classroom_panel_kw(ki, classrooms, weights)
        cloud_idx = int(demo_idx[ki])
        vox_bg = vox_prev if ki > 0 else None
        _append_classroom_pulse(panel_kw, vox_curr, vox_bg, cloud_idx)
        for k in range(n_step_sweep):
            u = smooth(float(k) / max(n_step_sweep - 1, 1))
            kw = _cloud_base(
                **_ch6_black_points_kw(n_pts, [cloud_idx]),
                **panel_kw,
            )
            kw.update(_ch6_voxel_sweep_layer(
                vox_curr, full_bounds,
                sweep_u=u,
                prev=vox_bg,
                blend_u=u,
                wave_amp=wave_amp,
                reveal=vox_bg is not None,
                uniform_alpha=vox_uniform,
            ))
            frames.append(_emit(**kw))
        vox_prev = vox_curr
        frames.extend(_hold(frames[-1], _workshop_hold(n_hold)))

    final_panel = _ch6_all_classrooms_ghost_panel_kw(classrooms, weights, mu=s["mu"])
    n_final_sweep = _workshop_short(64, 24)
    for k in range(n_final_sweep):
        u = smooth(float(k) / max(n_final_sweep - 1, 1))
        kw = _cloud_base(**final_panel)
        kw.update(_ch6_voxel_sweep_layer(
            vox_avg, full_bounds,
            sweep_u=u,
            prev=vox_prev,
            blend_u=u,
            wave_amp=wave_amp,
            reveal=True,
            uniform_alpha=vox_uniform,
        ))
        frames.append(_emit(**kw))
    frames.extend(_hold(frames[-1], _workshop_hold(n_hold * 2)))
    return frames


def _build_ch6_145_voxel_average_ten_classrooms_workshop(clip_id):
    """10-step classroom average (1→10) with black-point beats + wave sweeps."""
    return _build_ch6_voxel_average_classrooms_workshop(
        clip_id, n_demo=10, n_class=100, seed=141,
    )


def _build_ch6_147_voxel_average_ten_classrooms_n20_workshop(clip_id):
    """Same as ch6_145 but each classroom has n=20 students; 2D panel tracks roster."""
    return _build_ch6_voxel_average_classrooms_workshop(
        clip_id,
        n_demo=10,
        n_class=CH6_N_CLASS_BASE,
        seed=147,
        seed_from_key="D1",
        adapt_2d_classroom=True,
    )


def _build_ch6_148_voxel_average_ten_classrooms_n10_workshop(clip_id):
    """Same as ch6_147 but each classroom has n=10 students; 2D panel tracks roster."""
    return _build_ch6_voxel_average_classrooms_workshop(
        clip_id,
        n_demo=10,
        n_class=10,
        seed=148,
        adapt_2d_classroom=True,
    )


def _build_ch6_149_voxel_visit_ten_classrooms_workshop(clip_id):
    """Visit 10 classroom landscapes (n=100); average revealed at end."""
    return _build_ch6_voxel_visit_classrooms_workshop(
        clip_id, n_demo=10, n_class=100, seed=141,
    )


def _build_ch6_150_voxel_visit_ten_classrooms_n20_workshop(clip_id):
    """Visit 10 classroom landscapes (n=20); average revealed at end."""
    return _build_ch6_voxel_visit_classrooms_workshop(
        clip_id,
        n_demo=10,
        n_class=CH6_N_CLASS_BASE,
        seed=147,
        seed_from_key="D1",
    )


def _build_ch6_151_voxel_visit_ten_classrooms_n10_workshop(clip_id):
    """Visit 10 classroom landscapes (n=10); average revealed at end."""
    return _build_ch6_voxel_visit_classrooms_workshop(
        clip_id, n_demo=10, n_class=10, seed=148,
    )


def _build_ch6_146_voxel_classroom_grid_workshop(clip_id):
    """2×2 grid: each cell reveals a classroom dataset + diagonal voxel sweep."""
    scene = _ch6_movement_story_pack()
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    s = scene
    frames: list = []
    ax_lo, ax_hi = float(s["axis_lim"][0]), float(s["axis_lim"][1])
    full_bounds = (ax_lo, ax_hi, ax_lo, ax_hi, ax_lo, ax_hi)
    wave_amp = _ch6_nll_voxel_sweep_wave_amp()

    demo_idx = _ch6_pick_classroom_demo_indices(s["W"], s["w_class"], n=4, seed=146)
    classrooms = _ch6_classroom_roster_for_indices(scene, demo_idx)
    caches = [
        _ch6_nll_voxel_cache(cs, ce, cy, bounds=full_bounds)
        for cs, ce, cy in classrooms
    ]
    grid_ref = _ch6_duo_grid_reference_cell(scene)

    def _grid_frame(sweep_us, *, placeholders=4):
        cells = []
        for ci in range(4):
            if ci >= placeholders:
                cells.append(None)
            else:
                cs, ce, cy = classrooms[ci]
                cells.append(_ch6_render_classroom_grid_cell(
                    scene,
                    point_idx=int(demo_idx[ci]),
                    classroom=(cs, ce, cy),
                    bounds=full_bounds,
                    voxel_cache=caches[ci],
                    sweep_u=float(sweep_us[ci]),
                    wave_amp=wave_amp if float(sweep_us[ci]) < 1.0 - 1e-6 else 0.0,
                ))
        return _finish(
            _ch6_composite_2x2_classroom_cells(cells, ref_cell=grid_ref),
            clip_id,
        )

    n_sweep = _draft_short(52, 20)
    sweep_done = [0.0] * 4
    frames.extend(_hold(_grid_frame(sweep_done, placeholders=0), max(1, n_hold // 4)))

    for ci in range(4):
        for k in range(n_sweep):
            u = smooth(float(k) / max(n_sweep - 1, 1))
            sweep_done = [1.0] * 4
            sweep_done[ci] = u
            frames.append(_grid_frame(sweep_done, placeholders=ci + 1))
        frames.extend(_hold(_grid_frame([1.0] * 4, placeholders=4), max(1, n_hold // 3)))

    frames.extend(_hold(frames[-1], n_hold * 2))
    return frames


def _build_ch6_143_sandwich_theorem_restoring_push_from_142(clip_id):
    """Tutorial rails: restoring push + sandwich Cov(Δw) = H⁻¹ J H⁻¹."""
    scene = _ch6_movement_story_pack()
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    frames: list = []

    end_kw = dict(
        population_fade_u=1.0,
        marker_alpha_scale=0.06,
        grey_u=0.94,
        voxel_view_bounds=scene["zoom_bounds"],
        nll_voxel_cache=_ch6_movement_zoom_voxel_cache(scene),
        nll_voxel_sweep_u=1.0,
        nll_voxel_alpha_u=1.0,
        nll_voxel_uniform_alpha=float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA),
        highlight_classroom_w=scene["mu"],
        highlight_classroom_u=1.0,
        highlight_classroom_color="#111111",
        highlight_mu_u=1.0,
        highlight_mu_w=scene["mu"],
        mu_threshold_2d=scene["mu"],
        mu_threshold_2d_u=1.0,
        classroom_threshold_2d=scene["mu"],
        classroom_threshold_2d_u=1.0,
        knob_w=scene["mu"],
        ch6_pair_line_u=1.0,
        nll_grad_3d_u=0.0,
    )
    plot_end = _ch6_movement_plot_raw(scene, **end_kw)
    frames.extend(_hold(_finish(plot_end, clip_id), max(1, n_hold // 2)))

    for stage in range(5):
        bottom = _ch6_sandwich_bottom_blocks(stage)
        right = _ch6_sandwich_right_blocks(stage)
        stage_hold = n_hold if stage != 3 else n_hold * 2
        if stage == 0:
            _append_ch6_tutorial_morph(
                frames, clip_id, plot_end,
                scene=scene,
                n_layout=_draft_short(28, 12),
                n_panel=_draft_short(24, 10),
                bottom_blocks=bottom,
                right_blocks=right,
                bottom_title=CH6_MOVEMENT_SANDWICH_BOTTOM_TITLE,
                right_title=CH6_MOVEMENT_SANDWICH_RIGHT_TITLE,
                smooth=smooth,
                n_hold=n_hold,
            )
        else:
            shell_key = _ch6_movement_shell_key(right, sandwich=True, stage=stage)
            n_x = _draft_short(22, 9)
            for k in range(n_x):
                u = smooth(float(k) / max(n_x - 1, 1))
                if stage < 4:
                    right_prog = _ch6_block_write_prog_uniform(right, 1.0)
                else:
                    right_prog = _ch6_block_write_prog_frozen_lead(right, u, frozen_lead=1)
                frames.append(_ch6_compose_tutorial_frame(
                    plot_end,
                    layout_u=1.0,
                    panel_u=1.0,
                    title_u=1.0,
                    write_u=1.0,
                    bottom_blocks=bottom,
                    right_blocks=right,
                    bottom_prog=_ch6_block_write_prog_uniform(bottom, u),
                    right_prog=right_prog,
                    bottom_title=CH6_MOVEMENT_SANDWICH_BOTTOM_TITLE,
                    right_title=CH6_MOVEMENT_SANDWICH_RIGHT_TITLE,
                    right_title_single_line=True,
                    shell_cache_key=shell_key,
                ))
            frames.extend(_hold(frames[-1], max(1, stage_hold // 2)))

    frames.extend(_hold(frames[-1], n_hold * 2))
    return frames


def build_ch6_141_movement_formula_layout_from_133(clip_id):
    """Pre-push hold → tutorial layout; movement ≈ strength / decay."""
    return _build_ch6_141_movement_formula_layout_from_133(clip_id)


def build_ch6_142_classroom_voxel_average_reverse_push_from_141(clip_id):
    """Cloud voxel-average tour → classroom pulled toward average (black arrow)."""
    return _build_ch6_142_classroom_voxel_average_reverse_push_from_141(clip_id)


def build_ch6_144_voxel_average_wave_workshop(clip_id):
    """Workshop: diagonal wave sweeps while averaging classroom voxel landscapes."""
    return _build_ch6_144_voxel_average_wave_workshop(clip_id)


def build_ch6_145_voxel_average_ten_classrooms_workshop(clip_id):
    """10-classroom incremental average with black-point beats + wave sweeps."""
    return _build_ch6_145_voxel_average_ten_classrooms_workshop(clip_id)


def build_ch6_146_voxel_classroom_grid_workshop(clip_id):
    """2×2 grid of classroom datasets + voxel landscape sweeps."""
    return _build_ch6_146_voxel_classroom_grid_workshop(clip_id)


def build_ch6_147_voxel_average_ten_classrooms_n20_workshop(clip_id):
    """ch6_145 duplicate with n=20 classrooms; 2D panel tracks each roster."""
    return _build_ch6_147_voxel_average_ten_classrooms_n20_workshop(clip_id)


def build_ch6_148_voxel_average_ten_classrooms_n10_workshop(clip_id):
    """ch6_147 duplicate with n=10 classrooms; 2D panel tracks each roster."""
    return _build_ch6_148_voxel_average_ten_classrooms_n10_workshop(clip_id)


def build_ch6_149_voxel_visit_ten_classrooms_workshop(clip_id):
    """Visit 10 classroom landscapes; cumulative ghosts on 2D; average at end."""
    return _build_ch6_149_voxel_visit_ten_classrooms_workshop(clip_id)


def build_ch6_150_voxel_visit_ten_classrooms_n20_workshop(clip_id):
    """Visit 10 n=20 classroom landscapes; ghosts on 2D; average at end."""
    return _build_ch6_150_voxel_visit_ten_classrooms_n20_workshop(clip_id)


def build_ch6_151_voxel_visit_ten_classrooms_n10_workshop(clip_id):
    """Visit 10 n=10 classroom landscapes; ghosts on 2D; average at end."""
    return _build_ch6_151_voxel_visit_ten_classrooms_n10_workshop(clip_id)


def build_ch6_152_restoring_pull_slow_workshop(clip_id):
    """Eased classroom→average pull; slow deceleration, full pull arrow."""
    return _build_ch6_152_restoring_pull_slow_workshop(clip_id)


def build_ch6_153_restoring_pull_medium_workshop(clip_id):
    """Eased classroom→average pull; medium tempo."""
    return _build_ch6_153_restoring_pull_medium_workshop(clip_id)


def build_ch6_154_restoring_pull_soft_arrow_workshop(clip_id):
    """Eased pull with shorter, softer arrow."""
    return _build_ch6_154_restoring_pull_soft_arrow_workshop(clip_id)


def build_ch6_155_restoring_pull_glide_workshop(clip_id):
    """Very slow glide onto the average with fading pull arrow."""
    return _build_ch6_155_restoring_pull_glide_workshop(clip_id)


def build_ch6_156_classroom_avg_grad_variation_workshop(clip_id):
    """ch6_149 end → avg landscape → zoom → 50 classroom ∇NLL̄ beats."""
    return _build_ch6_156_classroom_avg_grad_variation_workshop(clip_id)


def build_ch6_157_classroom_avg_grad_variation_rotate_workshop(clip_id):
    """Same as ch6_156 with a full 360° spin across the gradient beat."""
    return _build_ch6_157_classroom_avg_grad_variation_rotate_workshop(clip_id)


def build_ch6_158_per_student_grad_fullscreen_workshop(clip_id):
    """From ch6_156 end: fullscreen 2D per-student ∇NLL (50 classrooms, reverse)."""
    return _build_ch6_158_per_student_grad_fullscreen_workshop(clip_id)


def build_ch6_159_per_student_grad_fullscreen_rotate_workshop(clip_id):
    """From ch6_157 end: fullscreen per-student grads after rotated beat."""
    return _build_ch6_159_per_student_grad_fullscreen_rotate_workshop(clip_id)


def build_ch6_160_sandwich_estimator_from_156_workshop(clip_id):
    """ch6_156 end → Ĥ_class + Ĵ → plug-in sandwich estimator."""
    return _build_ch6_160_sandwich_estimator_from_156_workshop(clip_id)


def build_ch6_143_sandwich_theorem_restoring_push_from_142(clip_id):
    """Restoring-pull view; derive Cov(Δw) = H⁻¹ J H⁻¹."""
    return _build_ch6_143_sandwich_theorem_restoring_push_from_142(clip_id)


def build_ch6_121_population_n100_covariance_matrix_arrows_from_120(clip_id):
    """From ch6_120 end: crossfade Hessian → sample Σ eigen-arrows, 270° orbit."""
    pack = _ch6_114_end_pack()
    end = _ch6_118_end_state(pack)
    info = end["info"]
    mu = end["mu"]
    W = np.asarray(end["W"], dtype=np.float64)
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    revealed = np.ones(len(W), dtype=bool)

    Xd = ch6_design(info["cs"], info["ce"])
    H = ch6_observed_information(mu, Xd, info["cy"], ridge=CH6_RIDGE)
    hess_dirs, hess_lens = _ch6_hessian_eigen_arrows(
        mu, H, box_bounds=end["box_bounds"],
    )
    C = np.cov(W.T, ddof=1)
    cov_dirs, cov_lens = _ch6_cov_eigen_arrows(
        mu, C, box_bounds=end["box_bounds"],
    )

    def _emit(azim, *, hess_u=0.0, cov_u=0.0):
        return _finish(_frame_population_variance_duo(
            xlim=info["xlim"],
            ylim=info["ylim"],
            base_study=info["cs"],
            base_exam=info["ce"],
            base_y=info["cy"],
            W=W,
            mu=mu,
            axis_lim=end["axis_lim"],
            revealed_mask=revealed,
            grey_u=0.0,
            show_center=True,
            range_state=end["full_ranges"],
            range_color=end["line_color"],
            show_box=False,
            view_elev=end["elev"],
            view_azim=float(azim),
            knob_w=info["w_knob"],
            ghost_fade_u=0.0,
            belief_z_lim=info.get("belief_z_lim"),
            mu_threshold_2d=info.get("mu_threshold_2d"),
            mu_threshold_2d_u=float(info.get("mu_threshold_2d_u", 1.0)),
            mu_threshold_2d_lw=3.6,
            marker_s=float(info.get("marker_s", 28)),
            marker_alpha_scale=1.0,
            hessian_arrow_dirs=hess_dirs,
            hessian_arrow_lengths=hess_lens,
            hessian_arrow_reveal_u=float(hess_u),
            cov_arrow_dirs=cov_dirs,
            cov_arrow_lengths=cov_lens,
            cov_arrow_reveal_u=float(cov_u),
        ), clip_id)

    az0 = float(end["azim"])
    n_spin = _draft_short(54, 22)
    n_xfade = max(1, n_spin // 3)

    frames: list = []
    frames.extend(_hold(_emit(az0, hess_u=1.0, cov_u=0.0), n_hold))

    for j in range(n_spin + 1):
        u = smooth(float(j) / max(n_spin, 1))
        az = az0 + 270.0 * u
        if j <= n_xfade:
            t = smooth(float(j) / max(n_xfade, 1))
            hess_u = 1.0 - t
            cov_u = t
        else:
            hess_u = 0.0
            cov_u = 1.0
        frames.append(_emit(az, hess_u=hess_u, cov_u=cov_u))

    frames.extend(_hold(_emit(az0 + 270.0, hess_u=0.0, cov_u=1.0), n_hold))
    return frames


def _build_ch6_grow_n7_to_100_param_cloud(
    clip_id, *, reel_pack, reel_label, n_reel_hq=300,
):
    """Shared: fullscreen n=7→100 growth, duo handoff, rotating parameter-cloud reel."""
    from ch6_frequentist import _CH3_DRAFT

    pack0 = _ch6_density_end_d1_pack()
    xlim, ylim = pack0["xlim"], pack0["ylim"]
    n_hold = CH6_N_HOLD * 2
    smooth = _g("ch3_knob_smoothstep")
    grad_span = 0.068
    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))

    def emit_fs(**kw):
        kw.setdefault("grad_span_frac", grad_span)
        kw.setdefault("grad_compact", True)
        kw.setdefault("show_point_grads_away_from_line", True)
        return _finish(_frame_ch6_fullscreen_2d_grad(xlim=xlim, ylim=ylim, **kw), clip_id)

    growth = _ch6_growth_sequence_to_n(target_n=100, seed=105)
    growth_ix = list(range(1, len(growth)))
    if _CH3_DRAFT:
        growth_ix = list(range(1, len(growth), 8)) + [len(growth) - 1]

    frames: list = []

    s7, e7, y7 = growth[0]
    w7, _ = ch6_fit_dataset(s7, e7, y7)
    frames.extend(_hold(emit_fs(
        base_study=s7, base_exam=e7, base_y=y7, w_live=w7,
    ), n_hold))

    for gi in growth_ix:
        s, e, y = growth[gi]
        w, _ = ch6_fit_dataset(s, e, y)
        frames.append(emit_fs(base_study=s, base_exam=e, base_y=y, w_live=w))
    s100, e100, y100 = growth[-1]
    w100, _ = ch6_fit_dataset(s100, e100, y100)
    frames.extend(_hold(frames[-1], n_hold))

    frames.extend(_hold(emit_fs(
        base_study=s100, base_exam=e100, base_y=y100, w_live=w100,
        show_point_grads_away_from_line=False,
    ), n_hold))

    n_layout_back = int(CH5_BEST_LINE_N_LAYOUT)
    for i in range(max(1, n_layout_back)):
        u = smooth(float(i) / max(n_layout_back - 1, 1))
        img = _frame_ch6_wide_to_duo_empty3d(
            u,
            xlim=xlim, ylim=ylim,
            base_study=s100, base_exam=e100, base_y=y100,
            w_live=w100, knob_w=w100, ghost_ws=[], ghost_fade_u=0.0,
            show_3d_u=0.0,
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], n_hold))

    for u in np.linspace(0.0, 1.0, _draft_short(10, 4)):
        img = _frame_ch6_wide_to_duo_empty3d(
            1.0,
            xlim=xlim, ylim=ylim,
            base_study=s100, base_exam=e100, base_y=y100,
            w_live=w100, knob_w=w100, ghost_ws=[], ghost_fade_u=0.0,
            show_3d_u=float(u),
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], n_hold // 2))

    n_reel = 24 if _CH3_DRAFT else int(n_reel_hq)
    print(f"  {reel_label}: building n=100 reel ({n_reel} classrooms)…", flush=True)
    reel_pack = reel_pack(int(n_reel))
    axis_lim = _ch6_marker_axis_lim_from_pack(reel_pack)
    pop_s, pop_e, pop_y = reel_pack["pop_s"], reel_pack["pop_e"], reel_pack["pop_y"]
    marker_s = ch6_population_n_sweep_marker_size(100)
    highlight_s = marker_s * (70.0 / 28.0)
    duo_kw = dict(show_legend=False, title_right=None)
    cloud_kw = dict(
        marker_z_mode="bias", marker_axis_lim=axis_lim, zlabel="b",
        **CH6_BELIEF_DENSITY_KW,
    )
    steps = reel_pack["reel_steps"]
    n_steps = len(steps)

    for ri, step in enumerate(steps):
        cs, ce, cy = step["study"], step["exam"], step["y"]
        w = step["w"]
        azim = base_azim + 360.0 * float(ri) / max(n_steps - 1, 1)
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y, pop_alpha=0.08,
            base_study=cs, base_exam=ce, base_y=cy,
            w_live=w, knob_w=w, highlight_w=w,
            ghost_ws=step["ghosts"],
            markers=step["markers"],
            marker_s=marker_s,
            highlight_marker_s=highlight_s,
            view_azim=azim,
            **cloud_kw,
            **duo_kw,
        )
        frames.append(_finish(img, clip_id))

    frames.extend(_hold(frames[-1], n_hold))
    print(f"  {reel_label}: done — {len(frames)} frames", flush=True)
    return frames


def _build_ch6_105_grow_n7_to_100_wild_param_cloud(clip_id):
    return _build_ch6_grow_n7_to_100_param_cloud(
        clip_id,
        reel_pack=_ch6_wild_n100_reel_pack_cached,
        reel_label="ch6_105",
    )


def build_ch6_105_grow_n7_to_100_wild_param_cloud(clip_id):
    return _build_ch6_105_grow_n7_to_100_wild_param_cloud(clip_id)


def _build_ch6_106_grow_n7_to_100_hetero_param_cloud(clip_id):
    return _build_ch6_grow_n7_to_100_param_cloud(
        clip_id,
        reel_pack=_ch6_heterogeneous_n100_reel_pack_cached,
        reel_label="ch6_106",
        n_reel_hq=600,
    )


def build_ch6_106_grow_n7_to_100_hetero_param_cloud(clip_id):
    return _build_ch6_106_grow_n7_to_100_hetero_param_cloud(clip_id)


def _build_ch6_101_pre_density_belief_color_spin(clip_id):
    """ch6_101 end-state cloud; belief-density colors reveal during 360° spin."""
    pack = _ch6_density_end_d1_pack()
    last = pack["reel_steps"][-1]
    cs, ce, cy = last["study"], last["exam"], last["y"]
    w = last["w"]
    markers = np.asarray(last["markers"], dtype=np.float64)
    ghosts = last["ghosts"]
    xlim, ylim = pack["xlim"], pack["ylim"]
    pop_s, pop_e, pop_y = pack["pop_s"], pack["pop_e"], pack["pop_y"]
    axis_lim = CH6_POPULATION_PARAM_AXIS_LIM
    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD * 2
    marker_s = ch6_population_n_sweep_marker_size(CH6_N_CLASS_BASE)
    highlight_s = marker_s * (70.0 / 28.0)
    n_spin = _draft_short(36, 14)

    def emit(reveal_u, azim):
        return _finish(_frame_duo(
            xlim=xlim, ylim=ylim,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y, pop_alpha=0.08,
            base_study=cs, base_exam=ce, base_y=cy,
            w_live=w, knob_w=w, highlight_w=w,
            ghost_ws=ghosts,
            markers=markers,
            marker_s=marker_s,
            highlight_marker_s=highlight_s,
            view_azim=azim,
            marker_z_mode="bias",
            marker_axis_lim=axis_lim,
            marker_density_reveal_u=float(reveal_u),
            ghost_density_reveal_u=float(reveal_u),
            zlabel="b",
            show_legend=False,
            title_left=None, title_right=None,
        ), clip_id)

    frames: list = []
    frames.extend(_hold(emit(0.0, base_azim), n_hold))
    for t in range(1, n_spin + 1):
        u = smooth(float(t) / float(n_spin))
        az = base_azim + 360.0 * float(t) / float(n_spin)
        frames.append(emit(u, az))
    frames.extend(_hold(frames[-1], n_hold))
    print(f"  {clip_id}: done — {len(frames)} frames", flush=True)
    return frames


def build_ch6_107_cloud_belief_density_color_spin(clip_id):
    return _build_ch6_101_pre_density_belief_color_spin(clip_id)


def build_ch6_108_cloud_belief_density_color_spin_ghosts(clip_id):
    return _build_ch6_101_pre_density_belief_color_spin(clip_id)


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


def _ch6_eased_pull_path_3d(w_start, w_target, n_frames, *, ease_power=2.5):
    """Monotone ease-out pull: classroom glides to average, slowing as it lands."""
    w_start = np.asarray(w_start, dtype=np.float64).reshape(3)
    w_target = np.asarray(w_target, dtype=np.float64).reshape(3)
    smooth = _g("ch3_knob_smoothstep")
    n = max(int(n_frames), 2)
    out: list[np.ndarray] = []
    ep = float(ease_power)
    for i in range(n):
        t = float(i) / max(n - 1, 1)
        u = 1.0 - (1.0 - t) ** ep
        u = smooth(u)
        out.append(w_start + u * (w_target - w_start))
    out[-1] = w_target.copy()
    return out


CH6_RESTORING_PULL_VARIANTS = {
    "slow": {
        "n_move": (80, 30),
        "ease_power": 3.8,
        "arrow_frac": 0.92,
        "arrow_u": 1.0,
        "hold_mult": 2,
    },
    "medium": {
        "n_move": (52, 20),
        "ease_power": 2.4,
        "arrow_frac": 0.78,
        "arrow_u": 1.0,
        "hold_mult": 1,
    },
    "soft_arrow": {
        "n_move": (84, 24),
        "ease_power": 3.0,
        "arrow_frac": 0.52,
        "arrow_u": 0.90,
        "hold_mult": 2,
    },
    "glide": {
        "n_move": (96, 34),
        "ease_power": 4.6,
        "arrow_frac": 0.88,
        "arrow_u": 0.82,
        "hold_mult": 3,
    },
}


def _build_ch6_restoring_pull_workshop(clip_id, *, variant: str = "slow"):
    """Workshop: ch6_142 zoomed landscape + eased classroom pull toward average."""
    cfg = CH6_RESTORING_PULL_VARIANTS[str(variant)]
    scene = _ch6_movement_story_pack()
    smooth = _g("ch3_knob_smoothstep")
    s = scene
    frames: list = []
    zoom_bounds = s["zoom_bounds"]
    vox_zoom = _ch6_movement_zoom_voxel_cache(scene)
    w_start = np.asarray(s["w_class"], dtype=np.float64).reshape(3)
    w_target = np.asarray(s["mu"], dtype=np.float64).reshape(3)
    delta = w_target - w_start
    dist0 = float(np.linalg.norm(delta))
    dir_fixed = delta / dist0 if dist0 > 1e-9 else np.zeros(3, dtype=np.float64)

    def _vox_layer(cache, bounds):
        return dict(
            voxel_view_bounds=bounds,
            nll_voxel_cache=cache,
            nll_voxel_sweep_u=1.0,
            nll_voxel_alpha_u=1.0,
            nll_voxel_uniform_alpha=float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA),
        )

    def _callouts(w_class_pt, *, avg_u=1.0, class_u=1.0):
        out = []
        if float(avg_u) > 1e-4:
            out.append(dict(
                point=s["mu"], label="average",
                color=CH6_VARIANCE_RED, u=float(avg_u),
                label_placement="bottom_outside",
            ))
        if float(class_u) > 1e-4:
            out.append(dict(
                point=np.asarray(w_class_pt, dtype=np.float64).reshape(3),
                label="this classroom",
                color="#111111", u=float(class_u),
                label_placement="top_outside",
            ))
        return out or None

    def _pull_kw(w_walk, *, arrow_len=0.0, arrow_u=1.0):
        ww = np.asarray(w_walk, dtype=np.float64).reshape(3)
        au = float(arrow_u)
        al = float(arrow_len)
        kw = dict(
            population_fade_u=1.0,
            marker_alpha_scale=0.0,
            grey_u=0.92,
            highlight_mu_u=1.0,
            highlight_mu_w=s["mu"],
            highlight_classroom_u=1.0,
            highlight_classroom_w=ww,
            highlight_classroom_color="#111111",
            ch6_pair_line_u=1.0,
            ch6_pair_line_alpha=1.0,
            knob_w=ww,
            mu_threshold_2d=s["mu"],
            mu_threshold_2d_u=1.0,
            ghost_threshold_2d=s["mu"],
            ghost_threshold_2d_u=1.0,
            classroom_threshold_2d=ww,
            classroom_threshold_2d_u=1.0,
            show_2d_grads=True,
            grad_w_live=ww,
            ch6_callouts=_callouts(ww, avg_u=1.0, class_u=1.0),
            **_vox_layer(vox_zoom, zoom_bounds),
        )
        if al > 1e-4 and au > 1e-4:
            kw.update(
                nll_grad_3d=dir_fixed,
                nll_grad_3d_w=ww,
                nll_grad_3d_scale=al,
                nll_grad_3d_u=au,
                nll_grad_3d_is_direction=True,
                nll_grad_3d_color="#111111",
                nll_grad_3d_axis_lim=s["grad_3d_axis_lim"],
            )
        return kw

    def _emit(**kw):
        return _finish(_ch6_movement_plot_raw(scene, **kw), clip_id)

    n_hold = _workshop_hold(CH6_N_HOLD * int(cfg["hold_mult"]))
    n_move = _workshop_short(int(cfg["n_move"][0]), int(cfg["n_move"][1]))
    arrow_frac = float(cfg["arrow_frac"])
    arrow_u_base = float(cfg["arrow_u"])
    ease_power = float(cfg["ease_power"])

    start_kw = _pull_kw(
        w_start,
        arrow_len=dist0 * arrow_frac,
        arrow_u=arrow_u_base,
    )
    frames.extend(_hold(_emit(**start_kw), n_hold))

    path = _ch6_eased_pull_path_3d(
        w_start, w_target, n_move, ease_power=ease_power,
    )
    for i, w_walk in enumerate(path[1:], start=1):
        t = float(i) / max(len(path) - 1, 1)
        remaining = float(np.linalg.norm(w_target - w_walk))
        arrow_len = remaining * arrow_frac
        fade = smooth(1.0 - 0.35 * t) if arrow_u_base < 1.0 else 1.0
        frames.append(_emit(**_pull_kw(
            w_walk,
            arrow_len=arrow_len,
            arrow_u=arrow_u_base * fade,
        )))

    end_kw = _pull_kw(w_target, arrow_len=0.0, arrow_u=0.0)
    frames.extend(_hold(_emit(**end_kw), n_hold))
    return frames


def _build_ch6_152_restoring_pull_slow_workshop(clip_id):
    return _build_ch6_restoring_pull_workshop(clip_id, variant="slow")


def _build_ch6_153_restoring_pull_medium_workshop(clip_id):
    return _build_ch6_restoring_pull_workshop(clip_id, variant="medium")


def _build_ch6_154_restoring_pull_soft_arrow_workshop(clip_id):
    return _build_ch6_restoring_pull_workshop(clip_id, variant="soft_arrow")


def _build_ch6_155_restoring_pull_glide_workshop(clip_id):
    return _build_ch6_restoring_pull_workshop(clip_id, variant="glide")


def _build_ch6_classroom_avg_grad_variation_workshop(
    clip_id,
    *,
    rotate_during_grad: bool = False,
):
    """From ch6_149 end: avg landscape label → zoom → 50 classroom ∇NLL̄ beats."""
    pack = _ch6_avg_grad_variation_pack(n_class=100, n_pick=50, n_demo=10, seed=141)
    s = pack["scene"]
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD
    frames: list = []
    n_pts = len(s["W"])
    full_bounds = pack["full_bounds"]
    zoom_bounds = pack["zoom_bounds"]
    wave_amp = _ch6_nll_voxel_sweep_wave_amp()
    vox_uniform = float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA)
    mu = np.asarray(s["mu"], dtype=np.float64).reshape(3)
    azim_base = float(s["azim_end"])
    vox_prev = pack["vox_each_visit"][-1] if pack["vox_each_visit"] else None
    vox_avg = pack["vox_avg_full"]
    final_panel = _ch6_all_classrooms_ghost_panel_kw(
        pack["visit_classrooms"], pack["visit_weights"], mu=mu,
    )

    def _emit(**kw):
        return _finish(_ch6_movement_plot_raw(s, **kw), clip_id)

    def _cloud_base(**extra):
        kw = dict(
            population_fade_u=1.0,
            marker_alpha_scale=0.78,
            grey_u=0.88,
            highlight_mu_u=0.65,
            highlight_mu_w=mu,
            highlight_classroom_u=0.45,
            ch6_pair_line_u=0.25,
            knob_w=mu,
            marker_below_voxels=True,
        )
        kw.update(extra)
        return kw

    def _vox_for_bounds(bounds):
        key = tuple(float(v) for v in bounds)
        hit = pack["vox_by_bounds"].get(key)
        if hit is None:
            hit = _ch6_cloud_workshop_voxel_cache_average(
                pack["avg_roster"], bounds=bounds, mu=mu, H=s["H_avg"],
            )
            pack["vox_by_bounds"][key] = hit
        return hit

    def _lerp_bounds(u, start_bounds, end_bounds):
        t = float(np.clip(u, 0.0, 1.0))
        return tuple(
            float(a) + t * (float(b) - float(a))
            for a, b in zip(start_bounds, end_bounds)
        )

    # --- Hold ch6_149 end state ---
    kw_end = _cloud_base(**final_panel)
    kw_end.update(_ch6_voxel_sweep_layer(
        vox_avg, full_bounds, sweep_u=1.0, prev=vox_prev,
        blend_u=1.0, wave_amp=0.0, reveal=True, uniform_alpha=vox_uniform,
    ))
    frames.extend(_hold(_emit(**kw_end), _workshop_hold(n_hold * 2)))

    # --- Average landscape sweep + label ---
    n_label_sweep = _workshop_short(52, 18)
    for k in range(n_label_sweep):
        u = smooth(float(k) / max(n_label_sweep - 1, 1))
        grow_u = smooth(max(0.0, (u - 0.04) / 0.96))
        label_u = 1.0 if u < 0.78 else max(0.0, 1.0 - (u - 0.78) / 0.22)
        kw = _cloud_base(
            **final_panel,
            title_3d="average landscape",
            title_3d_u=label_u,
            title_3d_scale=grow_u,
            title_3d_x=0.50,
            title_3d_y=0.56,
        )
        kw.update(_ch6_voxel_sweep_layer(
            vox_avg, full_bounds,
            sweep_u=u,
            prev=vox_prev,
            blend_u=u,
            wave_amp=wave_amp,
            reveal=True,
            uniform_alpha=vox_uniform,
        ))
        frames.append(_emit(**kw))
    frames.extend(_hold(_emit(**_cloud_base(**final_panel, **_ch6_voxel_sweep_layer(
        vox_avg, full_bounds, sweep_u=1.0, prev=vox_prev, blend_u=1.0,
        wave_amp=0.0, reveal=True, uniform_alpha=vox_uniform,
    ))), _workshop_hold(n_hold)))

    def _vox_layer_at(bounds, *, sweep_u=1.0):
        """Voxel grid rebuilt at ``bounds`` — constant cells/axis (ch6_142 zoom)."""
        bb = tuple(bounds)
        return _ch6_voxel_sweep_layer(
            _vox_for_bounds(bb), bb,
            sweep_u=float(sweep_u),
            wave_amp=0.0,
            uniform_alpha=vox_uniform,
        )

    # --- Zoom in: rebuild voxel grid at each window (ch6_142 texture consistency) ---
    n_zoom = _workshop_short(56, 22)
    print(f"  {clip_id}: precomputing {n_zoom} zoom voxel windows…", flush=True)
    zoom_windows: list = []
    for k in range(n_zoom):
        u = smooth(float(k) / max(n_zoom - 1, 1))
        b6 = _lerp_bounds(u, full_bounds, zoom_bounds)
        zoom_windows.append(b6)
        _vox_for_bounds(b6)
    _vox_for_bounds(zoom_bounds)
    for k, b6 in enumerate(zoom_windows):
        u = smooth(float(k) / max(n_zoom - 1, 1))
        panel_zoom = dict(final_panel)
        for _k in ("knob_w", "highlight_mu_w", "highlight_mu_u", "highlight_classroom_u"):
            panel_zoom.pop(_k, None)
        frames.append(_emit(
            population_fade_u=1.0,
            marker_alpha_scale=0.55 + 0.35 * u,
            grey_u=0.88 - 0.06 * u,
            marker_dim_scale=0.18,
            highlight_mu_u=0.70 + 0.25 * u,
            highlight_mu_w=mu,
            ch6_pair_line_u=0.25 + 0.55 * u,
            knob_w=mu,
            **panel_zoom,
            **_vox_layer_at(b6),
        ))
    zoom_vox_kw = _vox_layer_at(zoom_bounds)
    frames.extend(_hold(_emit(
        population_fade_u=1.0,
        marker_alpha_scale=0.90,
        grey_u=0.82,
        marker_dim_scale=0.16,
        highlight_mu_u=0.95,
        highlight_mu_w=mu,
        ch6_pair_line_u=0.80,
        knob_w=mu,
        **zoom_vox_kw,
    ), _workshop_hold(n_hold)))

    # --- Fade 2D visit ghosts / panel contents ---
    empty = np.array([], dtype=np.float64)
    n_fade_2d = _workshop_short(22, 8)
    zoom_kw = dict(
        population_fade_u=1.0,
        marker_alpha_scale=0.90,
        grey_u=0.82,
        marker_dim_scale=0.16,
        highlight_mu_u=0.95,
        highlight_mu_w=mu,
        ch6_pair_line_u=0.80,
        **zoom_vox_kw,
    )
    for k in range(n_fade_2d):
        u = smooth(float(k) / max(n_fade_2d - 1, 1))
        fade = 1.0 - u
        frames.append(_emit(
            base_study=empty, base_exam=empty, base_y=empty,
            ghost_classroom_rosters=pack["visit_classrooms"] if fade > 0.02 else [],
            ghost_ws=pack["visit_weights"] if fade > 0.02 else [],
            ghost_fade_u=fade,
            **zoom_kw,
        ))
    frames.extend(_hold(_emit(
        base_study=empty, base_exam=empty, base_y=empty,
        **zoom_kw,
    ), max(1, _workshop_hold(n_hold) // 2)))

    # --- Fast 50-classroom avg-gradient beat ---
    n_per = _workshop_short(3, 2)
    n_pick = len(pack["pick_idx"])
    azim_start = azim_base
    for ki, idx in enumerate(pack["pick_idx"]):
        cs, ce, cy = pack["picked_classrooms"][ki]
        w = pack["picked_weights"][ki]
        gdir = pack["grad_dirs"][ki]
        gscale = pack["grad_scales"][ki]
        if rotate_during_grad and n_pick > 1:
            u_rot = float(ki) / float(n_pick - 1)
            view_azim = azim_start + 360.0 * u_rot
        else:
            view_azim = azim_base
        panel_kw = _ch6_classroom_panel_kw(cs, ce, cy)
        panel_kw.update(
            mu_threshold_2d=mu,
            mu_threshold_2d_u=1.0,
            ghost_threshold_2d=mu,
            ghost_threshold_2d_u=1.0,
            classroom_threshold_2d=w,
            classroom_threshold_2d_u=1.0,
            classroom_threshold_2d_color="#111111",
        )
        beat_kw = {
            **zoom_kw,
            **panel_kw,
            **dict(
                highlight_classroom_u=1.0,
                highlight_classroom_color="#111111",
                highlight_mu_u=1.0,
                highlight_mu_w=mu,
                ch6_pair_line_u=1.0,
                view_azim=view_azim,
                nll_grad_3d=gdir,
                nll_grad_3d_w=w,
                nll_grad_3d_scale=gscale,
                nll_grad_3d_u=1.0,
                nll_grad_3d_is_direction=True,
                nll_grad_3d_color="#111111",
                nll_grad_3d_axis_lim=pack["grad_3d_axis_lim"],
            ),
        }
        beat_kw.update(_ch6_black_points_kw(n_pts, [int(idx)]))
        beat_frame = _emit(**beat_kw)
        frames.extend(_hold(beat_frame, n_per))
    frames.extend(_hold(frames[-1], _workshop_hold(n_hold * 2)))
    tag = "rotate" if rotate_during_grad else "static"
    print(f"  {clip_id}: avg-grad variation ({tag}) — {len(frames)} frames", flush=True)
    return frames


def _build_ch6_per_student_grad_fullscreen_workshop(
    clip_id,
    *,
    from_rotated_end: bool = False,
):
    """From ch6_156/157 end: fullscreen 2D + per-student ∇NLL (reverse 50 classrooms)."""
    pack = _ch6_avg_grad_variation_pack(n_class=100, n_pick=50, n_demo=10, seed=141)
    s = pack["scene"]
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD
    frames: list = []
    mu = np.asarray(s["mu"], dtype=np.float64).reshape(3)
    xlim, ylim = s["panel_xlim"], s["panel_ylim"]
    W = s["W"]
    n_pts = len(W)
    azim_base = float(s["azim_end"])
    azim_end = azim_base + (360.0 if from_rotated_end else 0.0)
    grad_span = 0.18
    empty = np.array([], dtype=np.float64)
    vox_uniform = float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA)
    zb = tuple(pack["zoom_bounds"])
    zoom_vox_kw = _ch6_voxel_sweep_layer(
        pack["vox_avg_zoom"], zb,
        sweep_u=1.0, uniform_alpha=vox_uniform,
    )
    zoom_kw = dict(
        population_fade_u=1.0,
        marker_alpha_scale=0.90,
        grey_u=0.82,
        marker_dim_scale=0.16,
        highlight_mu_u=0.95,
        highlight_mu_w=mu,
        ch6_pair_line_u=0.80,
        **zoom_vox_kw,
    )

    def _emit_duo(**kw):
        return _finish(_ch6_movement_plot_raw(s, **kw), clip_id)

    def _emit_fs(**kw):
        kw.setdefault("grad_span_frac", grad_span)
        kw.setdefault("grad_compact", False)
        return _finish(
            _frame_ch6_fullscreen_2d_grad(xlim=xlim, ylim=ylim, **kw),
            clip_id,
        )

    # --- Hold last frame of avg-grad beat ---
    last_i = len(pack["pick_idx"]) - 1
    idx = pack["pick_idx"][last_i]
    cs, ce, cy = pack["picked_classrooms"][last_i]
    w = pack["picked_weights"][last_i]
    panel_kw = _ch6_classroom_panel_kw(cs, ce, cy)
    panel_kw.update(
        mu_threshold_2d=mu, mu_threshold_2d_u=1.0,
        ghost_threshold_2d=mu, ghost_threshold_2d_u=1.0,
        classroom_threshold_2d=w, classroom_threshold_2d_u=1.0,
        classroom_threshold_2d_color="#111111",
    )
    end_kw = {
        **zoom_kw,
        **panel_kw,
        **dict(
            highlight_classroom_u=1.0,
            highlight_classroom_color="#111111",
            view_azim=azim_end,
            nll_grad_3d=pack["grad_dirs"][last_i],
            nll_grad_3d_w=w,
            nll_grad_3d_scale=pack["grad_scales"][last_i],
            nll_grad_3d_u=1.0,
            nll_grad_3d_is_direction=True,
            nll_grad_3d_color="#111111",
            nll_grad_3d_axis_lim=pack["grad_3d_axis_lim"],
        ),
    }
    end_kw.update(_ch6_black_points_kw(n_pts, [int(idx)]))
    frames.extend(_hold(_emit_duo(**end_kw), _workshop_hold(n_hold)))

    # --- Fade 3D voxels + cloud, keep 2D at duo size ---
    n_fade_vox = _workshop_short(16, 6)
    for k in range(n_fade_vox):
        u = smooth(float(k) / max(n_fade_vox - 1, 1))
        fade = 1.0 - u
        frames.append(_emit_duo(
            nll_voxel_alpha_u=fade,
            population_fade_u=fade,
            marker_alpha_scale=0.90 * fade,
            view_azim=azim_end,
            **panel_kw,
        ))

    # --- Resize 2D → fullscreen (ch6_103 pattern) ---
    def _morph_frame(layout_u, *, cloud_u=0.0):
        su = smooth(float(layout_u))
        return _finish(
            _frame_ch6_duo_cloud_layout(
                layout_u=su,
                cloud_u=float(cloud_u),
                knob_u=0.0,
                markers=W,
                highlight_w=w,
                view_azim=azim_end,
                marker_axis_lim=s["axis_lim"],
                marker_s=s["marker_s"],
                xlim=xlim, ylim=ylim,
                base_study=cs, base_exam=ce, base_y=cy,
                w_live=w,
                ghost_ws=[mu],
                ghost_fade_u=1.0,
            ),
            clip_id,
        )

    n_resize = _workshop_short(14, 6)
    for k in range(n_resize):
        u = float(k) / max(n_resize - 1, 1)
        frames.append(_morph_frame(u))
    frames.extend(_hold(_emit_fs(
        base_study=cs, base_exam=ce, base_y=cy, w_live=w,
        mu_threshold_2d=mu, classroom_threshold_2d=w,
        show_point_grads_away_from_line=True,
    ), _workshop_hold(n_hold)))

    # --- Reverse 50 classrooms: per-student grads + fixed thresholds ---
    n_per = _workshop_short(4, 2)
    for ki in range(len(pack["pick_idx"]) - 1, -1, -1):
        cs, ce, cy = pack["picked_classrooms"][ki]
        w = pack["picked_weights"][ki]
        fs_frame = _emit_fs(
            base_study=cs, base_exam=ce, base_y=cy,
            w_live=w,
            mu_threshold_2d=mu,
            classroom_threshold_2d=w,
            show_point_grads_away_from_line=True,
        )
        frames.extend(_hold(fs_frame, n_per))
    frames.extend(_hold(frames[-1], _workshop_hold(n_hold * 2)))
    print(f"  {clip_id}: per-student fullscreen grads — {len(frames)} frames", flush=True)
    return frames


def _build_ch6_160_sandwich_estimator_from_156_workshop(clip_id):
    """From ch6_156 end: Ĥ_class + Ĵ plug-ins → sandwich estimator Cov(Δw) ≈ Ĥ⁻¹ĴĤ⁻¹."""
    pack = _ch6_avg_grad_variation_pack(n_class=100, n_pick=50, n_demo=10, seed=141)
    s = pack["scene"]
    smooth = _g("ch3_knob_smoothstep")
    n_hold = CH6_N_HOLD
    frames: list = []
    n_pts = len(s["W"])
    mu = np.asarray(s["mu"], dtype=np.float64).reshape(3)
    vox_uniform = float(CH6_VOXEL_MEDIUM_UNIFORM_ALPHA)

    def _emit(**kw):
        return _finish(_ch6_movement_plot_raw(s, **kw), clip_id)

    end_kw, w_feat, cs, ce, cy = _ch6_avg_grad_156_end_kw(pack, s, n_pts=n_pts)
    zb = tuple(pack["zoom_bounds"])
    box_bounds = (
        (float(zb[0]), float(zb[1])),
        (float(zb[2]), float(zb[3])),
        (float(zb[4]), float(zb[5])),
    )
    Xd = ch6_design(cs, ce)
    H_class = ch6_observed_information(w_feat, Xd, cy, ridge=CH6_RIDGE)
    hess_dirs, hess_lens = _ch6_hessian_eigen_arrows(
        w_feat, H_class, box_bounds=box_bounds, length_frac=0.52,
    )
    last_i = len(pack["pick_idx"]) - 1
    gdir = pack["grad_dirs"][last_i]
    gscale = pack["grad_scales"][last_i]
    grad_tip = w_feat + gdir * gscale
    hess_tip = _ch6_hessian_arrow_tip(w_feat, hess_dirs, hess_lens)
    vox_avg_zoom = pack["vox_avg_zoom"]
    vox_class_zoom = _ch6_cloud_workshop_voxel_cache(cs, ce, cy, bounds=zb)
    wave_amp = _ch6_nll_voxel_sweep_wave_amp()

    def _call_kw(*, avg_u=1.0, class_u=1.0, grad_u=0.0, hess_u=0.0, **extra):
        kw = dict(
            ch6_callouts=_ch6_sandwich_160_callouts(
                mu, w_feat,
                avg_u=avg_u, class_u=class_u,
                grad_tip=grad_tip, grad_u=grad_u,
                hess_tip=hess_tip, hess_u=hess_u,
            ),
        )
        kw.update(extra)
        return kw

    class_vox_kw = _ch6_voxel_sweep_layer(
        vox_class_zoom, zb, sweep_u=1.0, uniform_alpha=vox_uniform,
    )

    # --- Hold ch6_156 end, then fade in average + classroom (opposite sides) ---
    frames.extend(_hold(_emit(**end_kw), _workshop_hold(n_hold)))
    n_call = _workshop_short(20, 8)
    for k in range(n_call):
        u = smooth(float(k) / max(n_call - 1, 1))
        frames.append(_emit(**end_kw, **_call_kw(avg_u=u, class_u=u)))
    frames.extend(_hold(_emit(**end_kw, **_call_kw(avg_u=1.0, class_u=1.0)), max(1, n_hold // 2)))

    # --- Landscape reveal: dim point labels, sweep avg→class, title at top ---
    n_land = _workshop_short(44, 16)
    for k in range(n_land):
        u = smooth(float(k) / max(n_land - 1, 1))
        grow_u = smooth(max(0.0, (u - 0.06) / 0.94))
        dim = max(0.0, 1.0 - 1.35 * u)
        kw = dict(end_kw)
        kw.update(_ch6_voxel_sweep_layer(
            vox_class_zoom, zb,
            sweep_u=u, prev=vox_avg_zoom, blend_u=u,
            wave_amp=wave_amp, reveal=True, uniform_alpha=vox_uniform,
        ))
        kw.update(
            title_3d="this classroom's landscape",
            title_3d_u=grow_u,
            title_3d_scale=grow_u,
            title_3d_x=0.50,
            title_3d_y=0.94,
            title_3d_va="top",
            title_3d_color=CH6_ESTIMATOR_LABEL_COLOR,
            nll_grad_3d_color=CH6_COV_COLOR,
            **_call_kw(avg_u=dim, class_u=dim),
        )
        frames.append(_emit(**kw))
    base_viz = dict(end_kw)
    base_viz.update(class_vox_kw)
    base_viz["nll_grad_3d_color"] = CH6_COV_COLOR
    xlim, ylim = s["panel_xlim"], s["panel_ylim"]
    call_full = dict(avg_u=1.0, class_u=1.0)
    frames.extend(_hold(_emit(**base_viz, **_call_kw(**call_full)), _workshop_hold(n_hold)))

    # --- Focus: avg ∇NLL on the arrow (hide point labels to reduce clutter) ---
    n_grad_lbl = _workshop_short(14, 6)
    for k in range(n_grad_lbl):
        u = smooth(float(k) / max(n_grad_lbl - 1, 1))
        frames.append(_emit(**base_viz, **_call_kw(avg_u=0.0, class_u=0.0, grad_u=u)))
    frames.extend(_hold(_emit(**base_viz, **_call_kw(
        avg_u=0.0, class_u=0.0, grad_u=1.0,
    )), max(1, n_hold // 2)))

    # --- Crossfade ∇NLL → Ĥ_class at the classroom weight ---
    n_fade_g = _workshop_short(20, 8)
    for k in range(n_fade_g):
        u = smooth(float(k) / max(n_fade_g - 1, 1))
        frames.append(_emit(**{
            **base_viz,
            **_call_kw(
                avg_u=0.25 * u,
                class_u=0.25 * u,
                grad_u=max(0.0, 1.0 - u),
                hess_u=u,
            ),
            "nll_grad_3d_u": 1.0 - u,
            "hessian_arrow_dirs": hess_dirs,
            "hessian_arrow_lengths": hess_lens,
            "hessian_arrow_reveal_u": u,
            "hessian_arrow_w": w_feat,
            "hessian_arrow_color": "#111111",
            "hessian_arrow_lw": 2.8,
            "hessian_arrow_axis_lim": pack["grad_3d_axis_lim"],
        }))
    hess_kw = dict(
        nll_grad_3d_u=0.0,
        hessian_arrow_dirs=hess_dirs,
        hessian_arrow_lengths=hess_lens,
        hessian_arrow_reveal_u=1.0,
        hessian_arrow_w=w_feat,
        hessian_arrow_color="#111111",
        hessian_arrow_lw=2.8,
        hessian_arrow_axis_lim=pack["grad_3d_axis_lim"],
    )
    frames.extend(_hold(_emit(**{
        **base_viz, **hess_kw, **_call_kw(**call_full, hess_u=1.0),
    }), _workshop_hold(n_hold)))

    # --- Per-student ∇ on 2D: soften 3D Ĥ label, grow 2D callout ---
    def _data_call(u, span_frac):
        return _ch6_sandwich_160_data_callout(
            cs, ce, cy, w_feat, u=u, span_frac=float(span_frac),
            xlim=xlim, ylim=ylim,
        )

    grad_span_mult = 2.0
    n_grad2d = _workshop_short(22, 9)
    for k in range(n_grad2d):
        u = smooth(float(k) / max(n_grad2d - 1, 1))
        span = grad_span_mult * (0.10 + 0.06 * u)
        dc = _data_call(u, span)
        frames.append(_emit(**{
            **base_viz, **hess_kw,
            **_call_kw(**call_full, hess_u=max(0.2, 1.0 - 0.55 * u)),
            "show_2d_grads": True,
            "grad_w_live": w_feat,
            "grad_span_frac": span,
            "hessian_arrow_reveal_u": max(0.30, 1.0 - 0.30 * u),
            "data_callouts": [dc] if dc is not None else None,
        }))
    data_call = _data_call(1.0, grad_span_mult * 0.16)
    plot_kw = {
        **base_viz, **hess_kw,
        **_call_kw(**call_full, hess_u=0.2),
        "show_2d_grads": True,
        "grad_w_live": w_feat,
        "grad_span_frac": grad_span_mult * 0.16,
        "hessian_arrow_reveal_u": 0.30,
        "data_callouts": [data_call] if data_call is not None else None,
    }
    plot_end = _ch6_movement_plot_raw(s, **plot_kw)
    frames.extend(_hold(_emit(**plot_kw), _workshop_hold(n_hold * 2)))

    # --- Tutorial layout + equation substitutions (true → estimated) ---
    n_stages = 4
    for stage in range(n_stages):
        bottom = _ch6_sandwich_estimator_bottom_blocks(stage)
        right = _ch6_sandwich_estimator_right_blocks(stage)
        stage_hold = n_hold * 2 if stage == n_stages - 1 else n_hold
        shell_key = f"{CH6_SANDWICH_ESTIMATOR_SHELL_KEY}:s{stage}"
        if stage == 0:
            _append_ch6_tutorial_morph(
                frames, clip_id, plot_end,
                scene=s,
                n_layout=_workshop_short(28, 11),
                n_panel=_workshop_short(24, 10),
                bottom_blocks=bottom,
                right_blocks=right,
                bottom_title=CH6_SANDWICH_ESTIMATOR_BOTTOM_TITLE,
                right_title=CH6_SANDWICH_ESTIMATOR_RIGHT_TITLE,
                smooth=smooth,
                n_hold=n_hold,
            )
        else:
            n_x = _workshop_short(20, 8)
            for k in range(n_x):
                u = smooth(float(k) / max(n_x - 1, 1))
                right_prog = (
                    _ch6_block_write_prog_frozen_lead(right, u, frozen_lead=1)
                    if stage >= 2
                    else _ch6_block_write_prog_uniform(right, u)
                )
                frames.append(_ch6_compose_tutorial_frame(
                    plot_end,
                    layout_u=1.0,
                    panel_u=1.0,
                    title_u=1.0,
                    write_u=1.0,
                    bottom_blocks=bottom,
                    right_blocks=right,
                    bottom_prog=_ch6_block_write_prog_uniform(bottom, u),
                    right_prog=right_prog,
                    bottom_title=CH6_SANDWICH_ESTIMATOR_BOTTOM_TITLE,
                    right_title=CH6_SANDWICH_ESTIMATOR_RIGHT_TITLE,
                    right_title_single_line=True,
                    shell_cache_key=shell_key,
                ))
            frames.extend(_hold(frames[-1], max(1, stage_hold // 2)))

    frames.extend(_hold(frames[-1], _workshop_hold(n_hold * 2)))
    print(f"  {clip_id}: sandwich estimator — {len(frames)} frames", flush=True)
    return frames


def _build_ch6_156_classroom_avg_grad_variation_workshop(clip_id):
    return _build_ch6_classroom_avg_grad_variation_workshop(
        clip_id, rotate_during_grad=False,
    )


def _build_ch6_157_classroom_avg_grad_variation_rotate_workshop(clip_id):
    return _build_ch6_classroom_avg_grad_variation_workshop(
        clip_id, rotate_during_grad=True,
    )


def _build_ch6_158_per_student_grad_fullscreen_workshop(clip_id):
    return _build_ch6_per_student_grad_fullscreen_workshop(
        clip_id, from_rotated_end=False,
    )


def _build_ch6_159_per_student_grad_fullscreen_rotate_workshop(clip_id):
    return _build_ch6_per_student_grad_fullscreen_workshop(
        clip_id, from_rotated_end=True,
    )


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


def _ch6_damped_newton_path_3d(
    w0,
    w_target,
    Xd,
    y,
    *,
    n_frames,
    ridge=CH6_RIDGE,
    axis_lim=None,
    dt=0.018,
    damp=2.8,
    newton_gain=0.72,
    pull_gain=0.40,
    max_substeps=520,
):
    """Damped Newton steps in 3-D with spring-like settling; resample to ``n_frames``."""
    w0 = np.asarray(w0, dtype=np.float64).reshape(3)
    target = np.asarray(w_target, dtype=np.float64).reshape(3)
    lo = float(axis_lim[0]) if axis_lim is not None else -4.0
    hi = float(axis_lim[1]) if axis_lim is not None else 4.0

    w = w0.copy()
    v = np.zeros(3, dtype=np.float64)
    traj = [w.copy()]
    times = [0.0]
    t = 0.0
    eye3 = np.eye(3, dtype=np.float64)
    for _ in range(int(max_substeps)):
        g = ch6_nll_grad(w, Xd, y, ridge=ridge)
        H = ch6_observed_information(w, Xd, y, ridge=ridge) + 0.12 * eye3
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            gn = float(np.linalg.norm(g))
            step = -g / gn if gn > 1e-12 else np.zeros(3, dtype=np.float64)
        to_tgt = target - w
        a = float(newton_gain) * step + float(pull_gain) * to_tgt - float(damp) * v
        v = v + float(dt) * a
        w = w + float(dt) * v
        w = np.clip(w, lo, hi)
        t += float(dt)
        traj.append(w.copy())
        times.append(t)
        if float(np.linalg.norm(w - target)) < 0.015 and float(np.linalg.norm(v)) < 0.04:
            break
    traj.append(target.copy())
    times.append(t + float(dt))
    pts = np.asarray(traj, dtype=np.float64)
    ts = np.asarray(times, dtype=np.float64)
    n_out = max(int(n_frames), 2)
    t_query = np.linspace(0.0, float(ts[-1]), n_out)
    out = np.zeros((n_out, 3), dtype=np.float64)
    for d in range(3):
        out[:, d] = np.interp(t_query, ts, pts[:, d])
    out[-1] = target.copy()
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
    add("population_reel_cloud_d1_2d_only", build_ch6_97_population_reel_cloud_d1_2d_only)
    add("population_reel_d1_wide_to_duo", build_ch6_98_population_reel_d1_wide_to_duo)
    add("population_reel_d1_split_to_duo", build_ch6_99_population_reel_d1_split_to_duo)
    add("population_cloud_d1_rotate_warp", build_ch6_100_population_cloud_d1_rotate_warp)
    add(
        "population_reel_cloud_dist_d1_density_end_spin",
        build_ch6_101_population_reel_cloud_dist_d1_density_end_spin,
    )
    add("cloud_loo_down_n6_spring", build_ch6_102_cloud_loo_down_n6_spring)
    add("cloud_loo_down_n6_spring_grads", build_ch6_103_cloud_loo_down_n6_spring_grads)
    add(
        "population_n7_up_dense_cloud_spin",
        build_ch6_104_population_n7_up_dense_cloud_spin,
    )
    add(
        "grow_n7_to_100_wild_param_cloud",
        build_ch6_105_grow_n7_to_100_wild_param_cloud,
    )
    add(
        "grow_n7_to_100_hetero_param_cloud",
        build_ch6_106_grow_n7_to_100_hetero_param_cloud,
    )
    add(
        "cloud_belief_density_color_spin",
        build_ch6_107_cloud_belief_density_color_spin,
    )
    add(
        "cloud_belief_density_color_spin_ghosts",
        build_ch6_108_cloud_belief_density_color_spin_ghosts,
    )
    add(
        "population_cloud_d1_rotate_warp_box",
        build_ch6_109_population_cloud_d1_rotate_warp_box,
    )
    add(
        "population_cloud_d1_rotate_warp_box_plain",
        build_ch6_110_population_cloud_d1_rotate_warp_box_plain,
    )
    add(
        "population_cloud_d1_rotate_warp_box_plain_planar",
        build_ch6_111_population_cloud_d1_rotate_warp_box_plain_planar,
    )
    add(
        "population_n100_zoom_spin_from_104",
        build_ch6_112_population_n100_zoom_spin_from_104,
    )
    add(
        "population_n100_zoom_spin_from_104_voxel",
        build_ch6_113_population_n100_zoom_spin_from_104_voxel,
    )
    add(
        "population_n100_marginal_axes_from_112",
        build_ch6_114_population_n100_marginal_axes_from_112,
    )
    add(
        "population_n100_variance_from_114",
        build_ch6_115_population_n100_variance_from_114,
    )
    add(
        "population_n100_variance_signed_from_114",
        build_ch6_116_population_n100_variance_signed_from_114,
    )
    add(
        "population_n100_box_warp_from_115",
        build_ch6_117_population_n100_box_warp_from_115,
    )
    add(
        "population_n100_hessian_plane_arrows_from_117",
        build_ch6_118_population_n100_hessian_plane_arrows_from_117,
    )
    add(
        "population_n100_covariance_from_118",
        build_ch6_119_population_n100_covariance_from_118,
    )
    add(
        "population_n100_covariance_math_from_119",
        build_ch6_120_population_n100_covariance_math_from_119,
    )
    add(
        "population_n100_covariance_matrix_arrows_from_120",
        build_ch6_121_population_n100_covariance_matrix_arrows_from_120,
    )
    add(
        "population_n100_cov_matrix_contrast_from_120",
        build_ch6_122_population_n100_cov_matrix_contrast_from_120,
    )
    add(
        "population_n100_cov_matrix_shrink_spin_from_122",
        build_ch6_123_population_n100_cov_matrix_shrink_spin_from_122,
    )
    add(
        "population_n100_covariance_outro_from_120",
        build_ch6_124_population_n100_covariance_outro_from_120,
    )
    add(
        "population_n100_covariance_intro_from_120",
        build_ch6_125_population_n100_covariance_intro_from_120,
    )
    add(
        "population_n100_nll_voxel_landscape_from_123",
        build_ch6_126_population_n100_nll_voxel_landscape_from_123,
    )
    add(
        "population_n100_nll_voxel_landscape_classroom_from_123",
        build_ch6_127_population_n100_nll_voxel_landscape_classroom_from_123,
    )
    add(
        "population_n100_nll_voxel_ct_scan_from_127",
        build_ch6_128_population_n100_nll_voxel_ct_scan_from_127,
    )
    add(
        "population_n100_classroom_spring_from_127",
        build_ch6_129_population_n100_classroom_spring_from_127,
    )
    add(
        "population_n100_classroom_spring_voxel_trail_from_127",
        build_ch6_130_population_n100_classroom_spring_voxel_trail_from_127,
    )
    add(
        "population_n100_classroom_w12_push_from_127",
        build_ch6_131_population_n100_classroom_w12_push_from_127,
    )
    add(
        "population_n100_classroom_w12_curvature_contrast_from_131",
        build_ch6_132_population_n100_classroom_w12_curvature_contrast_from_131,
    )
    add(
        "population_n100_classroom_spring_voxel_resistance_from_129",
        build_ch6_133_population_n100_classroom_spring_voxel_resistance_from_129,
    )
    add(
        "voxel_medium_flat_current",
        build_ch6_134_voxel_medium_flat_current,
    )
    add(
        "voxel_medium_steep_current",
        build_ch6_135_voxel_medium_steep_current,
    )
    add(
        "voxel_medium_shrinking_current_classroom",
        build_ch6_136_voxel_medium_shrinking_current_classroom,
    )
    add(
        "population_n100_classroom_spring_voxel_resistance_orbit_from_133",
        build_ch6_137_population_n100_classroom_spring_voxel_resistance_orbit_from_133,
    )
    add(
        "voxel_medium_flat_current_orbit",
        build_ch6_138_voxel_medium_flat_current_orbit,
    )
    add(
        "voxel_medium_steep_current_orbit",
        build_ch6_139_voxel_medium_steep_current_orbit,
    )
    add(
        "voxel_medium_shrinking_current_classroom_orbit",
        build_ch6_140_voxel_medium_shrinking_current_classroom_orbit,
    )
    add(
        "movement_formula_layout_from_133",
        build_ch6_141_movement_formula_layout_from_133,
    )
    add(
        "classroom_voxel_average_reverse_push_from_141",
        build_ch6_142_classroom_voxel_average_reverse_push_from_141,
    )
    add(
        "sandwich_theorem_restoring_push_from_142",
        build_ch6_143_sandwich_theorem_restoring_push_from_142,
    )
    add(
        "voxel_average_wave_workshop",
        build_ch6_144_voxel_average_wave_workshop,
    )
    add(
        "voxel_average_ten_classrooms_workshop",
        build_ch6_145_voxel_average_ten_classrooms_workshop,
    )
    add(
        "voxel_classroom_grid_workshop",
        build_ch6_146_voxel_classroom_grid_workshop,
    )
    add(
        "voxel_average_ten_classrooms_n20_workshop",
        build_ch6_147_voxel_average_ten_classrooms_n20_workshop,
    )
    add(
        "voxel_average_ten_classrooms_n10_workshop",
        build_ch6_148_voxel_average_ten_classrooms_n10_workshop,
    )
    add(
        "voxel_visit_ten_classrooms_workshop",
        build_ch6_149_voxel_visit_ten_classrooms_workshop,
    )
    add(
        "voxel_visit_ten_classrooms_n20_workshop",
        build_ch6_150_voxel_visit_ten_classrooms_n20_workshop,
    )
    add(
        "voxel_visit_ten_classrooms_n10_workshop",
        build_ch6_151_voxel_visit_ten_classrooms_n10_workshop,
    )
    add(
        "restoring_pull_slow_workshop",
        build_ch6_152_restoring_pull_slow_workshop,
    )
    add(
        "restoring_pull_medium_workshop",
        build_ch6_153_restoring_pull_medium_workshop,
    )
    add(
        "restoring_pull_soft_arrow_workshop",
        build_ch6_154_restoring_pull_soft_arrow_workshop,
    )
    add(
        "restoring_pull_glide_workshop",
        build_ch6_155_restoring_pull_glide_workshop,
    )
    add(
        "classroom_avg_grad_variation_workshop",
        build_ch6_156_classroom_avg_grad_variation_workshop,
    )
    add(
        "classroom_avg_grad_variation_rotate_workshop",
        build_ch6_157_classroom_avg_grad_variation_rotate_workshop,
    )
    add(
        "per_student_grad_fullscreen_workshop",
        build_ch6_158_per_student_grad_fullscreen_workshop,
    )
    add(
        "per_student_grad_fullscreen_rotate_workshop",
        build_ch6_159_per_student_grad_fullscreen_rotate_workshop,
    )
    add(
        "sandwich_estimator_from_156_workshop",
        build_ch6_160_sandwich_estimator_from_156_workshop,
    )
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
    result = builder(clip_id)
    ms = CH6_BOWL_MS if "bowl_nudge" in fn else CH6_MS
    if fn in {
        "ch6_53_population_reel_cloud_n20.mp4",
        "ch6_57_population_reel_cloud_dist_n20.mp4",
        "ch6_65_population_reel_cloud_n20_keep.mp4",
        "ch6_69_population_reel_cloud_dist_n20_keep.mp4",
    } or "population_n_sweep_cloud" in fn:
        ms = max(55, int(CH6_MS * 0.7))
    if fn in {
        "ch6_89_population_reel_cloud_dist_d1_density_end.mp4",
        "ch6_101_population_reel_cloud_dist_d1_density_end_spin.mp4",
        "ch6_100_population_cloud_d1_rotate_warp.mp4",
        "ch6_102_cloud_loo_down_n6_spring.mp4",
        "ch6_103_cloud_loo_down_n6_spring_grads.mp4",
        "ch6_104_population_n7_up_dense_cloud_spin.mp4",
        "ch6_105_grow_n7_to_100_wild_param_cloud.mp4",
        "ch6_106_grow_n7_to_100_hetero_param_cloud.mp4",
        "ch6_107_cloud_belief_density_color_spin.mp4",
        "ch6_108_cloud_belief_density_color_spin_ghosts.mp4",
    } or "population_cloud_d1_rotate_warp" in fn or "cloud_loo_down_n6_spring" in fn:
        ms = max(40, int(CH6_MS * 0.45))
    if isinstance(result, (list, tuple)):
        frames = result
        _g("save_mp4")(frames, fn, duration=ms)
        n_frames = len(frames)
    else:
        n_frames = 0

        def _counted():
            nonlocal n_frames
            for fr in result:
                n_frames += 1
                yield fr

        _g("save_mp4")(_counted(), fn, duration=ms)
    path = _g("OUTPUT_DIR") / fn
    print("wrote", path, f"({n_frames} frames, ms={ms})")
    return path
