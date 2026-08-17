"""Chapter 6 — prior / posterior (w_ST, w_EL) landscape construction (ch4-style knob sweeps)."""
from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Iterator

import numpy as np

from ch5_core import (
    CH5_BELIEF_SURFACE_ALPHA,
    CH5_HQ_GRID_N_FOCUS_FADE,
    CH5_HQ_GRID_N_FOCUS_HOLD,
    CH5_HQ_GRID_N_FOCUS_OPEN,
    CH5_HQ_GRID_FOCUS_DIM_ALPHA,
    CH5_HQ_GRID_FOCUS_DIM_GREY,
    CH5_MAP_PERTURB_DW_EL,
    CH5_MAP_PERTURB_DW_ST,
    CH5_MAP_PERTURB_N_HOLD,
    CH5_MAP_PERTURB_N_KNOB,
    CH5_MAP_PERTURB_N_ROT,
    CH5_MAP_PERTURB_ROT_DEG,
    CH5_D4_ORIGIN_DW_EL,
    CH5_D4_ORIGIN_DW_ST,
    CH5_D4_ORIGIN_N_CMAP,
    CH5_D4_ORIGIN_N_FADE,
    CH5_D4_ORIGIN_N_GUIDE,
    CH5_D4_ORIGIN_N_HOLD,
    CH5_D4_ORIGIN_N_KNOB,
    CH5_GRID_2D_ORBIT_N,
    CH5_GRID_2D_ORBIT_R_MAX,
    CH5_GRID_2D_ORBIT_R_MIN,
    CH5_GRID_2D_GUIDE_N,
    CH5_GRID_2D_CAM_N,
    CH5_GRID_2D_CAM_TOP_ELEV,
    CH5_GRID_2D_D4_DW_EL,
    CH5_GRID_2D_D4_DW_ST,
    CH5_GRID_2D_D4_N_SWING,
    CH5_GRID_2D_ZOOM_N,
    CH5_GRID_2D_ZOOM_N_HOLD,
    CH5_GRID_2D_ZOOM_TARGET,
    CH5_GRID_MAP_ROT_DEG,
    CH5_GRID_MAP_ROT_N,
    CH5_GRID_MAP_ROT_N_HOLD,
    CH5_STEM_SURF_HIST_ALPHA,
    CH5_STEM_SURF_HIST_WIDTH,
    CH5_STEM_SURF_LINE_ALPHA,
    CH5_STEM_SURF_LINE_COLOR,
    CH5_STEM_SURF_LINE_WIDTH_FRAC,
    CH5_STEM_SURF_LINEWIDTH,
    CH5_STEM_SURF_N_GROW_PER,
    CH5_STEM_SURF_N_HOLD,
    CH5_STEM_SURF_N_MORPH,
    CH5_STEM_SURF_N_PILLARS,
    CH5_STEM_SURF_N_TIGHTEN,
    CH5_STEM_SURF_PILLAR_SEED,
    CH5_STEM_SURF_POINT_ALPHA,
    CH5_STEM_SURF_POINT_SIZE,
    CH5_STEM_SURF_REVEAL_ORIGIN,
    CH5_STEM_SURF_SIDE_ALPHA,
    CH5_STEM_SURF_SIDE_COLOR,
    CH5_STEM_SURF_SIDE_EDGE_COLOR,
    CH5_STEM_SURF_SIDE_EDGE_WIDTH,
    CH5_LL_OVERLAY_ALPHA,
    CH5_LL_OVERLAY_BELIEF_ALPHA,
    CH5_LL_OVERLAY_BEST_LABEL,
    CH5_LL_OVERLAY_BEST_LABEL_FIG,
    CH5_LL_OVERLAY_COLOR,
    CH5_LL_OVERLAY_HEIGHT_FRAC,
    CH5_LL_OVERLAY_N_HOLD,
    CH5_LL_OVERLAY_N_ORBIT,
    CH5_LL_OVERLAY_N_REVEAL,
    CH5_LL_OVERLAY_ORBIT_DEG,
    CH5_LL_OVERLAY_REVEAL_ORIGIN,
    CH5_LL_OVERLAY_W_LIM,
    CH5_HQ_GRID_N_ORBIT,
    CH5_HQ_GRID_N_SQUISH,
    CH5_HQ_GRID_N_ZOOM,
    CH5_HQ_GRID_N_ZOOM_HOLD,
    CH5_HQ_GRID_N_ZOOM_ORBIT,
    CH5_HQ_GRID_ORBIT_DEG,
    CH5_HQ_GRID_LAND_DPI,
    CH5_HQ_GRID_LAND_GRID,
    CH5_HQ_LAND_DPI,
    CH5_HQ_LAND_GRID,
    CH5_HQ_LAND_GRID_COARSE,
    CH5_HQ_LAND_GRID_FINE,
    CH5_HQ_LAND_N_FILL_LINES,
    CH5_HQ_LAND_N_FILL_TRACE,
    CH5_HQ_LAND_N_HOLD,
    CH5_HQ_LAND_N_KNOB,
    CH5_HQ_N_KNOB_ZERO,
    CH5_HQ_LAND_N_REVEAL,
    CH5_HQ_LAND_N_ROT,
    CH5_HQ_LAND_QUADRANT_FINE,
    CH5_KNOB_ZERO,
    CH5_KNOBS_UNSET_FRAME_KW,
    CH5_PRIOR_LAND_DPI,
    CH5_PRIOR_LAND_GRID,
    CH5_PRIOR_LAND_GRID_COARSE,
    CH5_PRIOR_LAND_GRID_FINE,
    CH5_PRIOR_LAND_N_FILL_LINES,
    CH5_PRIOR_LAND_N_FILL_TRACE,
    CH5_PRIOR_LAND_N_HOLD,
    CH5_PRIOR_LAND_N_KNOB,
    CH5_PRIOR_LAND_N_REVEAL,
    CH5_PRIOR_LAND_N_ROT,
    CH5_PRIOR_LAND_QUADRANT_FINE,
    CH5_PRIOR_LANDSCAPE_SIGMA,
    CH5_SURFACE_GRID_SPACING,
    CH5_VIEW_BOUNDS,
    CH5_W12_B_FIXED,
    ch5_belief_w12_pdf,
    ch5_belief_w12_pdf_trace,
    ch5_clip_belief_height,
    ch5_log_likelihood_grid,
    ch5_posterior_w12_pdf,
    ch5_prior_w12_log_flat,
    ch5_prior_w12_pdf,
    ch5_belief_landscape_z_lim,
    ch5_hq_land_elev,
    ch5_surface_grid_plot_kw,
    ch5_prior_w12_z_lim,
)
from ch5_datasets import CH5_DATASET_KEYS, CH5_STANDARD_XLIM, CH5_STANDARD_YLIM, ch5_plot_limits
from ch5_layout import (
    ch5_composite_2x2_focus,
    ch5_composite_2x2_quadrants,
    ch5_quadrant_zoom_frame,
    ch5_uniform_belief_facecolors,
    ch5_uniform_belief_heatmap_color,
    ch5_uniform_belief_rgba_at_pdf,
    ch5_uniform_belief_z_lim,
)

_G: dict[str, Any] = {}

_CH5_GRID_SLOTS = {"D1": (0, 0), "D2": (0, 1), "D3": (1, 0), "D4": (1, 1)}


def install(globals_dict: dict[str, Any]) -> None:
    global _G
    _G = globals_dict


def _g(name: str):
    return _G[name]


def _ch5_hq_land_elev() -> float:
    return ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV")))


@dataclass(frozen=True)
class Ch5LandscapeConfig:
    grid: int
    grid_coarse: int
    grid_fine: int
    quadrant_fine: bool
    n_knob: int
    n_rot: int
    n_reveal: int
    n_fill_lines: int
    n_fill_trace: int
    n_hold: int
    dpi: int
    n_knob_zero: int = 0


def ch5_prior_landscape_config(*, hq: bool = False) -> Ch5LandscapeConfig:
    if hq:
        return Ch5LandscapeConfig(
            grid=int(CH5_HQ_LAND_GRID),
            grid_coarse=int(CH5_HQ_LAND_GRID_COARSE),
            grid_fine=int(CH5_HQ_LAND_GRID_FINE),
            quadrant_fine=bool(CH5_HQ_LAND_QUADRANT_FINE),
            n_knob=int(CH5_HQ_LAND_N_KNOB),
            n_rot=int(CH5_HQ_LAND_N_ROT),
            n_reveal=int(CH5_HQ_LAND_N_REVEAL),
            n_fill_lines=int(CH5_HQ_LAND_N_FILL_LINES),
            n_fill_trace=int(CH5_HQ_LAND_N_FILL_TRACE),
            n_hold=int(CH5_HQ_LAND_N_HOLD),
            dpi=int(CH5_HQ_LAND_DPI),
            n_knob_zero=int(CH5_HQ_N_KNOB_ZERO),
        )
    return Ch5LandscapeConfig(
        grid=int(CH5_PRIOR_LAND_GRID),
        grid_coarse=int(CH5_PRIOR_LAND_GRID_COARSE),
        grid_fine=int(CH5_PRIOR_LAND_GRID_FINE),
        quadrant_fine=bool(CH5_PRIOR_LAND_QUADRANT_FINE),
        n_knob=int(CH5_PRIOR_LAND_N_KNOB),
        n_rot=int(CH5_PRIOR_LAND_N_ROT),
        n_reveal=int(CH5_PRIOR_LAND_N_REVEAL),
        n_fill_lines=int(CH5_PRIOR_LAND_N_FILL_LINES),
        n_fill_trace=int(CH5_PRIOR_LAND_N_FILL_TRACE),
        n_hold=int(CH5_PRIOR_LAND_N_HOLD),
        dpi=int(CH5_PRIOR_LAND_DPI),
        n_knob_zero=0,
    )


def ch5_grid_landscape_config() -> Ch5LandscapeConfig:
    """HQ config with a denser (w_ST, w_EL) mesh for 2×2 grid cell renders."""
    base = ch5_prior_landscape_config(hq=True)
    return replace(
        base,
        grid=int(CH5_HQ_GRID_LAND_GRID),
        dpi=int(CH5_HQ_GRID_LAND_DPI),
    )


@contextmanager
def _landscape_render_context(dpi: int):
    """Lower or raise DPI during frame rasterization."""
    key = "CH3_ANIM_DPI"
    old = _G.get(key)
    _G[key] = int(dpi)
    try:
        yield
    finally:
        if old is None:
            _G.pop(key, None)
        else:
            _G[key] = old


def ch5_prior_w12_mesh_pack(
    prior_kind: str,
    *,
    b: float,
    w1_lo: float,
    w1_hi: float,
    w2_lo: float,
    w2_hi: float,
    grid_n=None,
    grid_coarse=None,
    grid_fine=None,
    quadrant_fine=None,
    sigma=None,
):
    """Mesh + prior height field on (w_ST, w_EL) with ``b`` fixed."""
    cfg = ch5_prior_landscape_config(hq=False)
    qfine = cfg.quadrant_fine if quadrant_fine is None else bool(quadrant_fine)
    nc = int(cfg.grid_coarse if grid_coarse is None else grid_coarse)
    nf = int(cfg.grid_fine if grid_fine is None else grid_fine)
    if qfine:
        mid = 0.0
        g1 = _g("_ch3_lik_w12_axis_refined")(w1_lo, w1_hi, mid, nc, nf, fine_lo_half=False)
        g2 = _g("_ch3_lik_w12_axis_refined")(w2_lo, w2_hi, mid, nc, nf, fine_lo_half=True)
    else:
        gn = int(cfg.grid if grid_n is None else grid_n)
        g1 = np.linspace(float(w1_lo), float(w1_hi), gn, dtype=np.float64)
        g2 = np.linspace(float(w2_lo), float(w2_hi), gn, dtype=np.float64)
    w1m, w2m = np.meshgrid(g1, g2, indexing="ij")
    bf = np.full(w1m.size, float(b), dtype=np.float64)
    lp = ch5_prior_w12_log_flat(
        w1m.ravel(), w2m.ravel(), bf,
        kind=prior_kind, sigma=sigma,
    )
    z_pdf = ch5_belief_w12_pdf(
        lp.reshape(w1m.shape),
        w1_lo=float(w1_lo), w1_hi=float(w1_hi), w2_lo=float(w2_lo), w2_hi=float(w2_hi),
    )
    z_lim_use = ch5_prior_w12_z_lim(prior_kind, scope="prior")
    z = ch5_clip_belief_height(z_pdf, prior_kind=prior_kind, z_lim=z_lim_use)
    return {
        "W1m": w1m,
        "W2m": w2m,
        "Z": z,
        "Z_pdf": np.asarray(z_pdf, dtype=float),
        "log_z_max": float(np.nanmax(lp)),
        "w1_lo": float(w1_lo),
        "w1_hi": float(w1_hi),
        "w2_lo": float(w2_lo),
        "w2_hi": float(w2_hi),
        "prior_kind": str(prior_kind).lower(),
    }


def ch5_posterior_w12_mesh_pack(
    study,
    exam,
    y,
    *,
    prior_kind: str = "gaussian",
    b: float | None = None,
    w1_lo: float | None = None,
    w1_hi: float | None = None,
    w2_lo: float | None = None,
    w2_hi: float | None = None,
    config: Ch5LandscapeConfig | None = None,
    nll_fn=None,
    z_lim=None,
):
    """Mesh + posterior height field on (w_ST, w_EL) with ``b`` fixed."""
    cfg = ch5_prior_landscape_config(hq=True) if config is None else config
    dlo1, dhi1, dlo2, dhi2, _, _ = CH5_VIEW_BOUNDS
    w1_lo = float(dlo1 if w1_lo is None else w1_lo)
    w1_hi = float(dhi1 if w1_hi is None else w1_hi)
    w2_lo = float(dlo2 if w2_lo is None else w2_lo)
    w2_hi = float(dhi2 if w2_hi is None else w2_hi)
    b = float(CH5_W12_B_FIXED if b is None else b)
    pk = str(prior_kind).lower()
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    # Belief heatmaps need a uniform mesh; quadrant refinement skews face colors per quadrant.
    use_quadrant_fine = bool(cfg.quadrant_fine) and pk != "uniform"
    if use_quadrant_fine:
        mid = 0.0
        g1 = _g("_ch3_lik_w12_axis_refined")(
            w1_lo, w1_hi, mid, cfg.grid_coarse, cfg.grid_fine, fine_lo_half=False,
        )
        g2 = _g("_ch3_lik_w12_axis_refined")(
            w2_lo, w2_hi, mid, cfg.grid_coarse, cfg.grid_fine, fine_lo_half=True,
        )
    else:
        g1 = np.linspace(w1_lo, w1_hi, int(cfg.grid), dtype=np.float64)
        g2 = np.linspace(w2_lo, w2_hi, int(cfg.grid), dtype=np.float64)
    w1m, w2m = np.meshgrid(g1, g2, indexing="ij")
    bf = np.full(w1m.size, b, dtype=np.float64)
    lp = ch5_prior_w12_log_flat(w1m.ravel(), w2m.ravel(), bf, kind=pk)
    if study.size == 0:
        log_post = lp.reshape(w1m.shape)
        z_pdf = ch5_belief_w12_pdf(
            log_post, w1_lo=w1_lo, w1_hi=w1_hi, w2_lo=w2_lo, w2_hi=w2_hi,
        )
    else:
        if nll_fn is None:
            nll_fn = _g("_ch3_nll_sum_on_flat_grid")
        ll = ch5_log_likelihood_grid(study, exam, y, w1m.ravel(), w2m.ravel(), bf, nll_fn=nll_fn)
        log_post = (lp + ll).reshape(w1m.shape)
        z_pdf = ch5_belief_w12_pdf(
            log_post, w1_lo=w1_lo, w1_hi=w1_hi, w2_lo=w2_lo, w2_hi=w2_hi,
        )
    if z_lim is None:
        z_lim = ch5_prior_w12_z_lim(pk)
    z = ch5_clip_belief_height(z_pdf, prior_kind=pk, z_lim=z_lim)
    k = int(np.nanargmax(log_post))
    flat = log_post.ravel()
    return {
        "W1m": w1m,
        "W2m": w2m,
        "Z": z,
        "Z_pdf": np.asarray(z_pdf, dtype=float),
        "log_post": log_post.reshape(w1m.shape),
        "w1_lo": w1_lo,
        "w1_hi": w1_hi,
        "w2_lo": w2_lo,
        "w2_hi": w2_hi,
        "ws": float(w1m.ravel()[k]),
        "we": float(w2m.ravel()[k]),
        "bb": b,
        "marker_z": float(z.ravel()[k]),
    }


def _g(name: str):
    return _G[name]


_CH5_FRAME_STRIP_KEYS = frozenset({
    "here_annotation", "here_label", "here_label_fig", "here_text_color",
    "show_threshold", "threshold_ws", "threshold_we", "threshold_bb", "threshold_label",
    "threshold_legend_dark",
    "show_shadow_marker", "shadow_marker_ws", "shadow_marker_we", "shadow_marker_z",
    "show_shadow_threshold", "shadow_threshold_ws", "shadow_threshold_we", "shadow_threshold_bb",
    "show_curves", "marker", "marker_ws", "marker_we", "marker_z",
    "elev", "azim", "landscape_reveal", "landscape_reveal_origin",
    "z_lik_ref", "z_label", "flat_surface", "mesh_pack", "z_lim", "curves",
    "study", "exam", "y",
    "origin_guides", "sigma_Z", "data_xlim", "data_ylim", "zero_axis_guides_u",
    "overlay_surface", "vertical_sprout", "belief_stems",
})


def _ch5_strip_frame_kwargs(fk: dict) -> dict:
    """Remove kwargs passed explicitly to ``ch3_frame_lik_w12_3d`` (avoids duplicates)."""
    out = dict(fk)
    for k in _CH5_FRAME_STRIP_KEYS:
        out.pop(k, None)
    return out


def _ch5_maybe_gc(frames: list, *, every: int = 12) -> None:
    if frames and len(frames) % int(every) == 0:
        gc.collect()


def _ch5_render_cleanup() -> None:
    """Drop matplotlib figures after each HQ landscape rasterization."""
    import matplotlib.pyplot as plt
    plt.close("all")
    gc.collect()


def _ch5_lik_w12_frame_kwargs(frame_kwargs: dict | None) -> dict:
    """Filter kwargs for ``ch3_frame_lik_w12_3d``; always use ch4 labeled knobs."""
    import inspect

    fk = {} if frame_kwargs is None else dict(frame_kwargs)
    if "knob_pack" not in fk:
        pack_fn = _G.get("ch4_knob_asset_pack")
        if pack_fn is not None:
            fk["knob_pack"] = pack_fn()
    fk.setdefault("weight_axis_labels", True)
    fn = _G.get("ch3_frame_lik_w12_3d")
    if fn is None:
        return fk
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fk
    return {k: v for k, v in fk.items() if k in params}


@dataclass
class _LandscapeRenderStep:
    """One HQ landscape frame (hold may repeat the same step)."""
    ws: float
    we: float
    bb: float
    elev: float
    azim: float
    mesh_pack: dict
    z_lim: tuple[float, float]
    z_ref: float
    curves: list = field(default_factory=list)
    study: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    exam: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    y: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    emp: str | None = "st"
    lrev: float = 0.0
    show_curves: bool = True
    marker: bool = False
    landscape_reveal_origin: str = "lo_lo"
    knobs_zero: bool = False
    marker_ws: float | None = None
    marker_we: float | None = None
    marker_z: float | None = None
    here_annotation: bool = False
    here_label: str = "most plausible line"
    show_threshold: bool = False
    threshold_ws: float | None = None
    threshold_we: float | None = None
    threshold_bb: float | None = None
    threshold_label: str = "most plausible line"
    show_shadow_marker: bool = False
    shadow_marker_ws: float | None = None
    shadow_marker_we: float | None = None
    shadow_marker_z: float | None = None
    show_shadow_threshold: bool = False
    shadow_threshold_ws: float | None = None
    shadow_threshold_we: float | None = None
    shadow_threshold_bb: float | None = None
    hold: int = 1
    prior_kind: str = "uniform"
    z_color_lim: tuple[float, float] | None = None
    show_surface: bool = True
    surface_grid: bool = False
    squish_u: float = 0.0
    origin_guides: dict | None = None
    sigma_Z: Any = None
    show_colormap_panel: bool = False
    knob_scales: list | None = None
    data_xlim: tuple[float, float] | None = None
    data_ylim: tuple[float, float] | None = None
    panel_sigma_stg: Any = None
    panel_sigma_elg: Any = None
    zero_axis_guides_u: float = 0.0


def _ch5_uniform_surface_facecolors(
    mesh_pack: dict,
    *,
    z_lim: tuple[float, float],
    z_color_lim: tuple[float, float],
    lrev: float = 1.0,
    origin: str = "lo_hi",
    surface_alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Belief heatmap facecolors with optional diagonal reveal mask."""
    W1m = mesh_pack["W1m"]
    W2m = mesh_pack["W2m"]
    Z = mesh_pack["Z"]
    Zpdf = np.asarray(mesh_pack.get("Z_pdf", Z), dtype=float)
    Zplot = ch5_clip_belief_height(Z, prior_kind="uniform", z_lim=z_lim)
    al = float(CH5_BELIEF_SURFACE_ALPHA if surface_alpha is None else surface_alpha)
    fc = ch5_uniform_belief_facecolors(Zpdf, z_lim=z_color_lim, surface_alpha=al)
    if float(lrev) < 1.0 - 1e-6:
        diag = _g("ch3_lik_w12_facecolors_diag")(
            W1m, W2m, float(lrev),
            w1_lo=float(mesh_pack["w1_lo"]),
            w1_hi=float(mesh_pack["w1_hi"]),
            w2_lo=float(mesh_pack["w2_lo"]),
            w2_hi=float(mesh_pack["w2_hi"]),
            rgba=(1.0, 1.0, 1.0, 1.0),
            origin=str(origin),
        )
        fc = np.asarray(fc, dtype=float).copy()
        fc[..., 3] *= diag[..., 3]
    return fc, Zplot


def _ch5_pdf_at_mesh(mesh_pack: dict, w1, w2) -> float:
    z_at = _g("_ch3_lik_w12_z_at")
    Zpdf = mesh_pack.get("Z_pdf", mesh_pack["Z"])
    return float(z_at(mesh_pack["W1m"], mesh_pack["W2m"], Zpdf, float(w1), float(w2)))


def _ch5_uniform_surface_grid_kw(mesh_pack: dict) -> dict:
    """Dark-grey grid lines on belief surfaces (~0.25 in w_ST / w_EL)."""
    kw = ch5_surface_grid_plot_kw(mesh_pack["W1m"], mesh_pack["W2m"])
    return {
        "surface_grid_edgecolor": str(kw["edgecolor"]),
        "surface_grid_linewidth": float(kw["linewidth"]),
        "surface_grid_rstride": int(kw["rstride"]),
        "surface_grid_cstride": int(kw["cstride"]),
    }


def _ch5_uniform_colored_curves(
    mesh_pack: dict,
    curves: list,
    *,
    z_color_lim: tuple[float, float],
) -> list:
    """One colored polyline per trace (mean pdf → Ch4 heatmap color)."""
    lo, hi = float(z_color_lim[0]), float(z_color_lim[1])
    out: list = []
    for cw1, cw2, cz in curves:
        cw1 = np.asarray(cw1, dtype=float)
        cw2 = np.asarray(cw2, dtype=float)
        cz = np.asarray(cz, dtype=float)
        if cw1.size < 2:
            continue
        pdfs = [_ch5_pdf_at_mesh(mesh_pack, cw1[i], cw2[i]) for i in range(int(cw1.size))]
        pdf_mean = float(np.mean(pdfs)) if pdfs else 0.0
        ccol = ch5_uniform_belief_heatmap_color(pdf_mean, lo, hi)
        out.append((cw1, cw2, cz, ccol))
    return out


def _ch5_squish_ct_z_lim(z_lim: tuple[float, float], squish_u: float) -> tuple[float, float]:
    """Morph Belief z-limits toward the CT bias axis while the surface squishes to b=0."""
    su = float(np.clip(float(squish_u), 0.0, 1.0))
    z0_lo, z0_hi = float(z_lim[0]), float(z_lim[1])
    ct_lo = float(CH5_VIEW_BOUNDS[4])
    ct_hi = float(CH5_VIEW_BOUNDS[5])
    return ((1.0 - su) * z0_lo + su * ct_lo, (1.0 - su) * z0_hi + su * ct_hi)


def _ch5_frame_lik_w12_belief(
    study,
    exam,
    y,
    ws: float,
    we: float,
    bb: float,
    *,
    mesh_pack: dict,
    z_lim: tuple[float, float],
    prior_kind: str = "uniform",
    z_color_lim: tuple[float, float] | None = None,
    curves: list | None = None,
    elev: float,
    azim: float,
    landscape_reveal: float = 1.0,
    landscape_reveal_origin: str = "lo_hi",
    show_curves: bool = True,
    marker: bool = False,
    marker_ws: float | None = None,
    marker_we: float | None = None,
    marker_z: float | None = None,
    z_lik_ref: float | None = None,
    z_label: str = "Belief",
    here_annotation: bool = False,
    here_label: str = "most plausible line",
    show_threshold: bool = False,
    threshold_ws: float | None = None,
    threshold_we: float | None = None,
    threshold_bb: float | None = None,
    threshold_label: str = "most plausible line",
    threshold_legend_dark: bool = False,
    show_shadow_marker: bool = False,
    shadow_marker_ws: float | None = None,
    shadow_marker_we: float | None = None,
    shadow_marker_z: float | None = None,
    show_shadow_threshold: bool = False,
    shadow_threshold_ws: float | None = None,
    shadow_threshold_we: float | None = None,
    shadow_threshold_bb: float | None = None,
    show_surface: bool = True,
    surface_grid: bool = False,
    squish_u: float = 0.0,
    frame_kwargs: dict | None = None,
    origin_guides: dict | None = None,
    sigma_Z=None,
    show_colormap_panel: bool = False,
    data_xlim: tuple[float, float] | None = None,
    data_ylim: tuple[float, float] | None = None,
    zero_axis_guides_u: float = 0.0,
    overlay_surface: dict | None = None,
    belief_surface_alpha: float | None = None,
    vertical_sprout: dict | None = None,
    belief_stems: dict | None = None,
    here_text_color: str | None = None,
    here_label_fig=None,
    **knob_kw,
):
    """Render one belief landscape frame; uniform prior uses shared red gradient."""
    pk = str(prior_kind).lower()
    fk = _ch5_strip_frame_kwargs(_ch5_lik_w12_frame_kwargs(frame_kwargs))
    if show_threshold:
        fk.pop("show_legend", None)
    if show_colormap_panel or sigma_Z is not None:
        fk = dict(fk)
        fk["show_colormap"] = bool(show_colormap_panel or sigma_Z is not None)
    flat_surface = None
    lrev = float(landscape_reveal)
    curve_list = [] if curves is None else list(curves)
    zc = z_color_lim or ch5_uniform_belief_z_lim() if pk == "uniform" else None
    if pk == "uniform" and show_curves and curve_list:
        curve_list = _ch5_uniform_colored_curves(
            mesh_pack, curve_list, z_color_lim=zc,
        )
    if pk == "uniform" and show_surface:
        fc, Zplot = _ch5_uniform_surface_facecolors(
            mesh_pack,
            z_lim=z_lim,
            z_color_lim=zc,
            lrev=lrev,
            origin=landscape_reveal_origin,
            surface_alpha=belief_surface_alpha,
        )
        mz = marker_z
        if marker and mz is None and marker_ws is not None and marker_we is not None:
            mz = float(_g("_ch3_lik_w12_z_at")(
                mesh_pack["W1m"], mesh_pack["W2m"], Zplot, float(marker_ws), float(marker_we),
            ))
        flat_surface = {
            "W1m": mesh_pack["W1m"],
            "W2m": mesh_pack["W2m"],
            "Z": Zplot,
            "facecolors": fc,
            "marker_z": mz,
            "surface_grid": bool(surface_grid),
        }
        su = float(np.clip(float(squish_u), 0.0, 1.0))
        if su > 1e-6:
            # Compress toward Belief=0 while the display axis morphs to CT b∈[-3,3],
            # so the sheet settles at b=0 (correct CT height) as the axis resizes.
            flat_surface["Z"] = np.asarray(flat_surface["Z"], dtype=float) * (1.0 - su)
            if mz is not None:
                mz = float(mz) * (1.0 - su)
                flat_surface["marker_z"] = mz
                marker_z = mz
            z_lim = _ch5_squish_ct_z_lim(z_lim, su)
            if su >= 0.45:
                z_label = r"$b$"
        if surface_grid:
            flat_surface.update(_ch5_uniform_surface_grid_kw(mesh_pack))
        lrev = 0.0
    elif not show_surface:
        lrev = 0.0
    else:
        # Non-uniform / non-flat path: still morph axis if a squish is requested.
        su = float(np.clip(float(squish_u), 0.0, 1.0))
        if su > 1e-6:
            z_lim = _ch5_squish_ct_z_lim(z_lim, su)
            if su >= 0.45:
                z_label = r"$b$"
    try:
        call_kw = dict(fk)
        call_kw.update(knob_kw)
        if show_threshold:
            call_kw["show_legend"] = False
        # Force annotation / legend styling after fk merge (fk must not win).
        call_kw["here_text_color"] = here_text_color
        call_kw["here_label_fig"] = here_label_fig
        call_kw["threshold_legend_dark"] = bool(threshold_legend_dark)
        return _g("ch3_frame_lik_w12_3d")(
            study, exam, y, float(ws), float(we), float(bb),
            mesh_pack=mesh_pack,
            z_lim=z_lim,
            flat_surface=flat_surface,
            curves=curve_list,
            elev=float(elev),
            azim=float(azim),
            landscape_reveal=lrev,
            landscape_reveal_origin=landscape_reveal_origin,
            show_curves=show_curves,
            marker=marker,
            marker_ws=marker_ws,
            marker_we=marker_we,
            marker_z=marker_z,
            here_annotation=here_annotation,
            here_label=here_label,
            show_threshold=show_threshold,
            threshold_ws=threshold_ws,
            threshold_we=threshold_we,
            threshold_bb=threshold_bb,
            threshold_label=threshold_label,
            show_shadow_marker=show_shadow_marker,
            shadow_marker_ws=shadow_marker_ws,
            shadow_marker_we=shadow_marker_we,
            shadow_marker_z=shadow_marker_z,
            show_shadow_threshold=show_shadow_threshold,
            shadow_threshold_ws=shadow_threshold_ws,
            shadow_threshold_we=shadow_threshold_we,
            shadow_threshold_bb=shadow_threshold_bb,
            z_lik_ref=z_lik_ref,
            z_label=z_label,
            sigma_Z=sigma_Z,
            origin_guides=origin_guides,
            data_xlim=data_xlim,
            data_ylim=data_ylim,
            zero_axis_guides_u=float(zero_axis_guides_u),
            overlay_surface=overlay_surface,
            vertical_sprout=vertical_sprout,
            belief_stems=belief_stems,
            **call_kw,
        )
    finally:
        _ch5_render_cleanup()


def _ch5_render_landscape_step(step: _LandscapeRenderStep, frame_kwargs: dict | None) -> Any:
    wz, ez, bz = CH5_KNOB_ZERO
    ws, we, bb = step.ws, step.we, step.bb
    if step.knobs_zero:
        ws, we, bb = wz, ez, bz
        knob_kw = dict(CH5_KNOBS_UNSET_FRAME_KW)
    elif step.emp is None:
        # Keep dial angles; equal/unset emphasis (e.g. camera-only motion).
        knob_kw = dict(CH5_KNOBS_UNSET_FRAME_KW)
        if step.knob_scales is not None:
            knob_kw["knob_scales"] = list(step.knob_scales)
    else:
        knob_kw = {"emphasize_knob": step.emp}
        if step.knob_scales is not None:
            knob_kw["knob_scales"] = list(step.knob_scales)
    annot_kw = {"here_text_color": "white"} if step.here_annotation else {}
    fk_n = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    if step.show_threshold:
        fk_n.pop("show_legend", None)
        fk_n = _ch5_strip_frame_kwargs(fk_n)
    return _ch5_frame_lik_w12_belief(
        step.study, step.exam, step.y,
        ws, we, bb,
        mesh_pack=step.mesh_pack,
        z_lim=step.z_lim,
        prior_kind=step.prior_kind,
        z_color_lim=step.z_color_lim,
        curves=step.curves,
        elev=step.elev,
        azim=step.azim,
        landscape_reveal=step.lrev,
        landscape_reveal_origin=step.landscape_reveal_origin,
        show_curves=step.show_curves,
        marker=step.marker,
        marker_ws=step.marker_ws,
        marker_we=step.marker_we,
        marker_z=step.marker_z,
        here_annotation=step.here_annotation,
        here_label=step.here_label,
        show_threshold=step.show_threshold,
        threshold_ws=step.threshold_ws,
        threshold_we=step.threshold_we,
        threshold_bb=step.threshold_bb,
        threshold_label=step.threshold_label,
        show_shadow_marker=step.show_shadow_marker,
        shadow_marker_ws=step.shadow_marker_ws,
        shadow_marker_we=step.shadow_marker_we,
        shadow_marker_z=step.shadow_marker_z,
        show_shadow_threshold=step.show_shadow_threshold,
        shadow_threshold_ws=step.shadow_threshold_ws,
        shadow_threshold_we=step.shadow_threshold_we,
        shadow_threshold_bb=step.shadow_threshold_bb,
        z_lik_ref=step.z_ref,
        show_surface=step.show_surface,
        surface_grid=step.surface_grid,
        squish_u=step.squish_u,
        origin_guides=step.origin_guides,
        sigma_Z=step.sigma_Z,
        show_colormap_panel=step.show_colormap_panel,
        data_xlim=step.data_xlim,
        data_ylim=step.data_ylim,
        zero_axis_guides_u=step.zero_axis_guides_u,
        frame_kwargs=fk_n,
        **annot_kw,
        **knob_kw,
    )


def _ch5_prior_landscape_step_stream(
    prior_kind: str,
    cfg: Ch5LandscapeConfig,
    *,
    end_knobs_zero: bool = False,
    fixed_end_camera: bool = False,
) -> Iterator[_LandscapeRenderStep]:
    """Yield render steps for the prior knob-sweep / reveal choreography."""
    prior_kind = str(prior_kind).lower()
    sigma = float(CH5_PRIOR_LANDSCAPE_SIGMA)
    dlo1, dhi1, dlo2, dhi2, _, _ = CH5_VIEW_BOUNDS
    lo, hi = float(dlo1), float(dhi1)
    b0 = float(CH5_W12_B_FIXED)
    w1s, w2s = hi, lo
    wz, ez, bz = CH5_KNOB_ZERO
    if prior_kind == "uniform":
        z_color_lim = ch5_uniform_belief_z_lim()
    else:
        z_color_lim = None

    n_knob = int(cfg.n_knob)
    n_rot = int(cfg.n_rot)
    n_reveal = int(cfg.n_reveal)
    n_hold = int(cfg.n_hold)
    n_fill_lines = int(cfg.n_fill_lines)
    n_trace_fill = int(cfg.n_fill_trace)
    n_trace = max(12, n_knob)

    el_ct = _ch5_hq_land_elev()
    az_ct = float(_g("CH3_LIK_W12_CT_AZIM"))
    if fixed_end_camera:
        # Match ch5_58 fixed end-camera (no W1/W2 view dance).
        el_w1 = el_w2 = el_ct
        az_w1 = az_w2 = az_ct
    else:
        el_w1 = float(_g("CH3_LIK_W12_ELEV_W1"))
        az_w1 = float(_g("CH3_LIK_W12_AZIM_W1"))
        el_w2 = float(_g("CH3_LIK_W12_ELEV_W2"))
        az_w2 = float(_g("CH3_LIK_W12_AZIM_W2"))

    empty_study = np.array([], dtype=np.float64)
    empty_exam = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)
    curves: list = []

    if prior_kind == "uniform":
        z_lim_full = ch5_prior_w12_z_lim("uniform", scope="prior")
        z_color_lim = ch5_uniform_belief_z_lim()
    elif prior_kind == "gaussian":
        z_lim_full = ch5_prior_w12_z_lim("gaussian", scope="prior")
        z_color_lim = None
    else:
        z_lim_full = None
        z_color_lim = None
    mesh0 = ch5_posterior_w12_mesh_pack(
        empty_study, empty_exam, empty_y,
        prior_kind=prior_kind,
        b=b0,
        w1_lo=lo, w1_hi=hi, w2_lo=lo, w2_hi=hi,
        config=cfg,
        z_lim=z_lim_full,
    )
    z_ref = float(np.nanmax(mesh0["Z"]))
    if z_lim_full is None:
        z_lim_full = (0.0, max(z_ref * 1.06, 1e-9))

    def _step(
        ws, we, bb, *, elev, azim, emp="st", lrev=0.0, show_curves=True,
        marker=True, landscape_reveal_origin="lo_lo", knobs_zero=False,
        marker_ws=None, marker_we=None, hold=1,
        show_surface=False, surface_grid=False,
    ) -> _LandscapeRenderStep:
        return _LandscapeRenderStep(
            ws=float(ws), we=float(we), bb=float(bb),
            elev=float(elev), azim=float(azim),
            mesh_pack=mesh0,
            z_lim=z_lim_full,
            z_ref=z_ref,
            curves=list(curves),
            study=empty_study,
            exam=empty_exam,
            y=empty_y,
            emp=emp,
            lrev=float(lrev),
            show_curves=show_curves,
            marker=marker,
            landscape_reveal_origin=landscape_reveal_origin,
            knobs_zero=knobs_zero,
            marker_ws=marker_ws,
            marker_we=marker_we,
            hold=int(hold),
            prior_kind=prior_kind,
            z_color_lim=z_color_lim,
            show_surface=bool(show_surface),
            surface_grid=bool(surface_grid),
        )

    yield _step(wz, ez, bz, elev=el_w1, azim=az_w1, show_curves=False, marker=False, hold=n_hold)

    for tv in np.linspace(0.0, 1.0, n_knob, endpoint=True):
        u = _g("ch3_knob_smoothstep")(float(tv))
        w1c = _g("ch3_lerp")(hi, lo, u)
        cw1, cw2, cz = _ch5_prior_w12_trace_knob1(
            prior_kind, w2s, b0, hi, w1c, n_trace, sigma=sigma, grid_n=int(cfg.grid),
            mesh0=mesh0,
        )
        if curves:
            curves[-1] = (cw1, cw2, cz)
        else:
            curves.append((cw1, cw2, cz))
        yield _step(w1c, w2s, b0, elev=el_w1, azim=az_w1, emp="st", marker_ws=w1c, marker_we=w2s)
    yield _step(lo, w2s, b0, elev=el_w1, azim=az_w1, marker_ws=lo, marker_we=w2s, hold=max(1, n_hold // 2))
    curves[-1] = _ch5_prior_w12_trace_knob1(
        prior_kind, w2s, b0, hi, lo, n_trace, sigma=sigma, grid_n=int(cfg.grid),
        mesh0=mesh0,
    )

    for tv in np.linspace(0.0, 1.0, n_knob, endpoint=True):
        u = _g("ch3_knob_smoothstep")(float(tv))
        w1c = _g("ch3_lerp")(lo, hi, u)
        yield _step(w1c, w2s, b0, elev=el_w1, azim=az_w1, emp="st", marker_ws=w1c, marker_we=w2s)
    yield _step(w1s, w2s, b0, elev=el_w1, azim=az_w1, marker_ws=w1s, marker_we=w2s, hold=max(1, n_hold // 2))

    for tv in np.linspace(0.0, 1.0, n_rot, endpoint=True):
        u = _g("ch3_knob_smoothstep")(float(tv))
        yield _step(
            w1s, w2s, b0,
            elev=_g("ch3_lerp")(el_w1, el_w2, u),
            azim=_g("_ch3_lik_w12_lerp_azim_shortest")(az_w1, az_w2, u),
            knobs_zero=True, marker=False,
        )

    curves.append(_ch5_prior_w12_trace_knob2(
        prior_kind, w1s, b0, w2s, w2s, 2, sigma=sigma, grid_n=int(cfg.grid),
        mesh0=mesh0,
    ))
    for tv in np.linspace(0.0, 1.0, n_knob, endpoint=True):
        u = _g("ch3_knob_smoothstep")(float(tv))
        w2c = _g("ch3_lerp")(w2s, hi, u)
        cw1, cw2, cz = _ch5_prior_w12_trace_knob2(
            prior_kind, w1s, b0, w2s, w2c, n_trace, sigma=sigma, grid_n=int(cfg.grid),
            mesh0=mesh0,
        )
        curves[-1] = (cw1, cw2, cz)
        yield _step(w1s, w2c, b0, elev=el_w2, azim=az_w2, emp="el", marker_ws=w1s, marker_we=w2c)
    yield _step(w1s, hi, b0, elev=el_w2, azim=az_w2, marker_ws=w1s, marker_we=hi, hold=max(1, n_hold // 2))

    for tv in np.linspace(0.0, 1.0, n_knob, endpoint=True):
        u = _g("ch3_knob_smoothstep")(float(tv))
        w2c = _g("ch3_lerp")(hi, w2s, u)
        yield _step(w1s, w2c, b0, elev=el_w2, azim=az_w2, emp="el", marker_ws=w1s, marker_we=w2c)
    yield _step(w1s, w2s, b0, elev=el_w2, azim=az_w2, marker_ws=w1s, marker_we=w2s, hold=max(1, n_hold // 2))

    for tv in np.linspace(0.0, 1.0, n_rot, endpoint=True):
        u = _g("ch3_knob_smoothstep")(float(tv))
        yield _step(
            w1s, w2s, b0,
            elev=_g("ch3_lerp")(el_w2, el_ct, u),
            azim=_g("_ch3_lik_w12_lerp_azim_shortest")(az_w2, az_ct, u),
            knobs_zero=True, marker=False,
        )

    w1_vals = np.linspace(lo, hi, n_fill_lines, dtype=np.float64)
    w2_vals = np.linspace(lo, hi, n_fill_lines, dtype=np.float64)
    for w1v in w1_vals:
        cw1, cw2, cz = _ch5_prior_w12_trace_knob2(
            prior_kind, float(w1v), b0, lo, hi, n_trace_fill, sigma=sigma, grid_n=int(cfg.grid),
            mesh0=mesh0,
        )
        curves.append((cw1, cw2, cz))
        yield _step(w1s, w2s, b0, elev=el_ct, azim=az_ct, emp="st", marker=False, knobs_zero=True)
    for w2v in w2_vals:
        cw1, cw2, cz = _ch5_prior_w12_trace_knob1(
            prior_kind, float(w2v), b0, lo, hi, n_trace_fill, sigma=sigma, grid_n=int(cfg.grid),
            mesh0=mesh0,
        )
        curves.append((cw1, cw2, cz))
        yield _step(w1s, w2s, b0, elev=el_ct, azim=az_ct, emp="el", marker=False, knobs_zero=True)

    yield _step(w1s, w2s, b0, elev=el_ct, azim=az_ct, marker=False, knobs_zero=True, hold=max(1, n_hold // 2))

    # Curves complete — reveal full colored surface (no diagonal wipe); drop trace lines.
    use_grid = prior_kind == "uniform"
    yield _step(
        w1s, w2s, b0, elev=el_ct, azim=az_ct, lrev=1.0,
        show_curves=False, marker=False, knobs_zero=True,
        show_surface=True, surface_grid=use_grid,
        landscape_reveal_origin="lo_hi",
    )
    yield _step(
        w1s, w2s, b0, elev=el_ct, azim=az_ct, lrev=1.0,
        show_curves=False, marker=False, knobs_zero=True,
        show_surface=True, surface_grid=use_grid,
        landscape_reveal_origin="lo_hi",
        hold=n_hold,
    )

    if end_knobs_zero and int(cfg.n_knob_zero) > 1:
        for tv in np.linspace(0.0, 1.0, int(cfg.n_knob_zero), endpoint=True):
            u = _g("ch3_knob_smoothstep")(float(tv))
            ws = _g("ch3_lerp")(w1s, wz, u)
            we = _g("ch3_lerp")(w2s, ez, u)
            bb = _g("ch3_lerp")(b0, bz, u)
            yield _step(
                ws, we, bb, elev=el_ct, azim=az_ct, lrev=1.0,
                show_curves=False, marker=False, landscape_reveal_origin="lo_hi",
                knobs_zero=True, show_surface=True, surface_grid=use_grid,
            )


def _ch5_prior_w12_trace_heights(mesh0, w1s, w2s) -> np.ndarray:
    """Sample trace z from the prior mesh so curve heights match the finished surface."""
    z_at = _g("_ch3_lik_w12_z_at")
    W1m = np.asarray(mesh0["W1m"], dtype=float)
    W2m = np.asarray(mesh0["W2m"], dtype=float)
    Z = np.asarray(mesh0["Z"], dtype=float)
    w1a = np.asarray(w1s, dtype=float)
    w2a = np.asarray(w2s, dtype=float)
    return np.array(
        [float(z_at(W1m, W2m, Z, w1a[i], w2a[i])) for i in range(w1a.size)],
        dtype=float,
    )


def _ch5_prior_w12_trace_knob1(
    prior_kind, w2_fix, b, w1_from, w1_to, n, *, sigma=None, grid_n=52, mesh0=None,
):
    w1s = np.linspace(float(w1_from), float(w1_to), int(n), dtype=float)
    w2s = np.full_like(w1s, float(w2_fix))
    if mesh0 is not None:
        z = _ch5_prior_w12_trace_heights(mesh0, w1s, w2s)
    else:
        bf = np.full_like(w1s, float(b))
        lp = ch5_prior_w12_log_flat(w1s, w2s, bf, kind=prior_kind, sigma=sigma)
        dlo1, dhi1, dlo2, dhi2, _, _ = CH5_VIEW_BOUNDS
        z = ch5_belief_w12_pdf_trace(
            lp, w1_lo=dlo1, w1_hi=dhi1, w2_lo=dlo2, w2_hi=dhi2, grid_n=grid_n,
        )
    return w1s, w2s, np.asarray(z, dtype=float)


def _ch5_prior_w12_trace_knob2(
    prior_kind, w1_fix, b, w2_from, w2_to, n, *, sigma=None, grid_n=52, mesh0=None,
):
    w2s = np.linspace(float(w2_from), float(w2_to), int(n), dtype=float)
    w1s = np.full_like(w2s, float(w1_fix))
    if mesh0 is not None:
        z = _ch5_prior_w12_trace_heights(mesh0, w1s, w2s)
    else:
        bf = np.full_like(w2s, float(b))
        lp = ch5_prior_w12_log_flat(w1s, w2s, bf, kind=prior_kind, sigma=sigma)
        dlo1, dhi1, dlo2, dhi2, _, _ = CH5_VIEW_BOUNDS
        z = ch5_belief_w12_pdf_trace(
            lp, w1_lo=dlo1, w1_hi=dhi1, w2_lo=dlo2, w2_hi=dhi2, grid_n=grid_n,
        )
    return w1s, w2s, np.asarray(z, dtype=float)


def _ch5_knob_zero_tail(
    frames,
    *,
    ws_end,
    we_end,
    bb_end,
    mesh0,
    z_lim_full,
    z_ref,
    el_ct,
    az_ct,
    frame_kwargs,
    n_knob_zero,
    prior_kind: str = "uniform",
    z_color_lim=None,
):
    if int(n_knob_zero) <= 1:
        return
    wz, ez, bz = CH5_KNOB_ZERO
    empty_study = np.array([], dtype=np.float64)
    empty_exam = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)
    for tv in np.linspace(0.0, 1.0, int(n_knob_zero), endpoint=True):
        u = _g("ch3_knob_smoothstep")(float(tv))
        ws = _g("ch3_lerp")(float(ws_end), wz, u)
        we = _g("ch3_lerp")(float(we_end), ez, u)
        bb = _g("ch3_lerp")(float(bb_end), bz, u)
        frames.append(_ch5_frame_lik_w12_belief(
            empty_study, empty_exam, empty_y,
            ws, we, bb,
            mesh_pack=mesh0,
            z_lim=z_lim_full,
            prior_kind=prior_kind,
            z_color_lim=z_color_lim,
            curves=[],
            elev=el_ct,
            azim=az_ct,
            landscape_reveal=1.0,
            landscape_reveal_origin="lo_hi",
            show_curves=False,
            marker=False,
            z_lik_ref=z_ref,
            z_label="Belief",
            frame_kwargs=frame_kwargs,
            **CH5_KNOBS_UNSET_FRAME_KW,
        ))


def ch5_build_prior_w12_landscape_frames(
    prior_kind: str,
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    end_knobs_zero: bool = False,
    knobs_zero_on_rotate: bool = False,
    fixed_end_camera: bool = False,
) -> list:
    """
    Knob-sweep construction of the prior surface over (w_ST, w_EL), b fixed.

    Uses ``CH5_PRIOR_LAND_*`` preview settings (override with CH5_PRIOR_LANDSCAPE_FULL=1).
    Pass ``config=ch5_prior_landscape_config(hq=True)`` for high-resolution exports.
    ``fixed_end_camera=True`` keeps elev/azim at the ch5_58 end pose throughout.
    """
    del knobs_zero_on_rotate  # reserved; rotate steps already pass knobs_zero where needed
    prior_kind = str(prior_kind).lower()
    cfg = ch5_prior_landscape_config(hq=False) if config is None else config
    fk = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    frames: list = []
    with _landscape_render_context(cfg.dpi):
        for step in _ch5_prior_landscape_step_stream(
            prior_kind, cfg,
            end_knobs_zero=end_knobs_zero,
            fixed_end_camera=bool(fixed_end_camera),
        ):
            fr = _ch5_render_landscape_step(step, fk)
            for _ in range(max(1, int(step.hold))):
                frames.append(fr)
    return frames


def ch5_build_sequential_posterior_w12_frames(
    study,
    exam,
    y,
    point_order,
    *,
    prior_kind: str = "gaussian",
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    annotate_final: bool = True,
    n_seq_hold: int = 1,
    n_annot_hold: int = 0,
    n_orbit: int = 0,
) -> list:
    """Reveal D1 points one-by-one; 3D landscape tracks the posterior at each step."""
    cfg = ch5_prior_landscape_config(hq=True) if config is None else config
    fk = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    pk = str(prior_kind).lower()
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    order = [int(j) for j in point_order]
    wz, ez, bz = CH5_KNOB_ZERO

    el_ct = _ch5_hq_land_elev()
    az_ct = float(_g("CH3_LIK_W12_CT_AZIM"))

    frames: list = []
    with _landscape_render_context(cfg.dpi):
        empty_study = np.array([], dtype=np.float64)
        empty_exam = np.array([], dtype=np.float64)
        empty_y = np.array([], dtype=np.int64)
        z_color_lim = ch5_uniform_belief_z_lim() if pk == "uniform" else None

        def _subset(n_show: int):
            idxs = order[: int(n_show)]
            return study[idxs], exam[idxs], y[idxs]

        n_pts = len(order)
        orbit_state = None
        for n in range(0, n_pts + 1):
            phase_u = 0.0 if n_pts == 0 else float(n) / float(n_pts)
            z_lim_n = ch5_belief_landscape_z_lim(pk, phase_u=phase_u)
            sn, en, yn = _subset(n)
            mesh_n = ch5_posterior_w12_mesh_pack(
                sn, en, yn, prior_kind=pk, config=cfg, z_lim=z_lim_n,
            )
            z_ref_n = float(np.nanmax(mesh_n["Z"]))
            final = bool(annotate_final and n == n_pts)
            mws = float(mesh_n["ws"]) if final else None
            mwe = float(mesh_n["we"]) if final else None
            mbb = float(mesh_n["bb"]) if final else None
            mz = float(mesh_n["marker_z"]) if final else None
            fk_n = _ch5_lik_w12_frame_kwargs(fk)
            if final:
                fk_n.pop("show_legend", None)
                fk_n = _ch5_strip_frame_kwargs(fk_n)
            annot_kw = {"here_text_color": "white"} if final else {}
            use_grid = pk == "uniform"
            fr = _ch5_frame_lik_w12_belief(
                sn, en, yn, wz, ez, bz,
                mesh_pack=mesh_n,
                z_lim=z_lim_n,
                prior_kind=pk,
                z_color_lim=z_color_lim,
                curves=[],
                elev=el_ct,
                azim=az_ct,
                landscape_reveal=1.0,
                landscape_reveal_origin="lo_hi",
                show_curves=False,
                marker=final,
                marker_ws=mws,
                marker_we=mwe,
                marker_z=mz,
                here_annotation=final,
                here_label="most plausible line",
                show_threshold=final,
                threshold_ws=mws,
                threshold_we=mwe,
                threshold_bb=mbb,
                threshold_label="most plausible line",
                z_lik_ref=z_ref_n,
                show_surface=True,
                surface_grid=use_grid,
                frame_kwargs=fk_n,
                **annot_kw,
                **CH5_KNOBS_UNSET_FRAME_KW,
            )
            hold_n = int(cfg.n_hold if n == 0 else n_seq_hold)
            for _ in range(max(1, hold_n)):
                frames.append(fr)
                _ch5_maybe_gc(frames)
            if final:
                orbit_state = dict(
                    sn=sn, en=en, yn=yn,
                    mesh_n=mesh_n, z_lim_n=z_lim_n, z_ref_n=z_ref_n,
                    mws=mws, mwe=mwe, mbb=mbb, mz=mz,
                    fk_n=fk_n, use_grid=use_grid,
                )

        if annotate_final and int(n_annot_hold) > 0:
            for _ in range(int(n_annot_hold)):
                frames.append(frames[-1])

        n_spin = int(n_orbit)
        if orbit_state is not None and n_spin > 0:
            orbit_deg = float(CH5_HQ_GRID_ORBIT_DEG)
            os = orbit_state
            # endpoint=True so the last frame lands on az_ct + 360° (full turn).
            for tv in np.linspace(0.0, 1.0, n_spin, endpoint=True):
                az_spin = float(az_ct) + orbit_deg * float(tv)
                frames.append(_ch5_frame_lik_w12_belief(
                    os["sn"], os["en"], os["yn"], wz, ez, bz,
                    mesh_pack=os["mesh_n"],
                    z_lim=os["z_lim_n"],
                    prior_kind=pk,
                    z_color_lim=z_color_lim,
                    curves=[],
                    elev=el_ct,
                    azim=az_spin,
                    landscape_reveal=1.0,
                    landscape_reveal_origin="lo_hi",
                    show_curves=False,
                    marker=True,
                    marker_ws=os["mws"],
                    marker_we=os["mwe"],
                    marker_z=os["mz"],
                    here_annotation=True,
                    here_label="most plausible line",
                    show_threshold=True,
                    threshold_ws=os["mws"],
                    threshold_we=os["mwe"],
                    threshold_bb=os["mbb"],
                    threshold_label="most plausible line",
                    z_lik_ref=os["z_ref_n"],
                    show_surface=True,
                    surface_grid=os["use_grid"],
                    frame_kwargs=os["fk_n"],
                    here_text_color="white",
                    **CH5_KNOBS_UNSET_FRAME_KW,
                ))
                _ch5_maybe_gc(frames)

    return frames


def _ch5_render_landscape_grid_cells(
    step: _LandscapeRenderStep,
    datasets: dict[str, dict],
    visible_keys: set[str],
    frame_kwargs: dict | None,
    *,
    per_key: dict[str, dict] | None = None,
) -> list[list]:
    """Render one HQ landscape cell per visible dataset (2×2 array, not composited)."""
    cells = [[None, None], [None, None]]
    pk = step.prior_kind
    z_color_lim = step.z_color_lim or (
        ch5_uniform_belief_z_lim() if pk == "uniform" else ch5_prior_w12_z_lim(pk, scope="global")
    )
    for key in CH5_DATASET_KEYS:
        if key not in visible_keys:
            continue
        i, j = _CH5_GRID_SLOTS[key]
        ds = datasets[key]
        extra = {} if per_key is None else dict(per_key.get(key, {}))
        mesh_pack = extra.get("mesh_pack", step.mesh_pack)
        z_lim = extra.get("z_lim", step.z_lim)
        z_ref = extra.get("z_ref", step.z_ref)
        study = extra.get("study", step.study)
        exam = extra.get("exam", step.exam)
        y_arr = extra.get("y", step.y)
        final = bool(extra.get("final", False))
        mws = extra.get("marker_ws") if final else None
        mwe = extra.get("marker_we") if final else None
        mz = extra.get("marker_z") if final else None
        cell_step = _LandscapeRenderStep(
            ws=step.ws,
            we=step.we,
            bb=step.bb,
            elev=step.elev,
            azim=step.azim,
            mesh_pack=mesh_pack,
            z_lim=z_lim,
            z_ref=z_ref,
            curves=step.curves,
            study=study,
            exam=exam,
            y=y_arr,
            emp=step.emp,
            lrev=step.lrev,
            show_curves=step.show_curves,
            marker=final,
            landscape_reveal_origin=step.landscape_reveal_origin,
            knobs_zero=step.knobs_zero,
            marker_ws=mws,
            marker_we=mwe,
            marker_z=mz,
            here_annotation=bool(extra.get("here_annotation", final)),
            here_label=step.here_label,
            show_threshold=final,
            threshold_ws=extra.get("threshold_ws") if final else None,
            threshold_we=extra.get("threshold_we") if final else None,
            threshold_bb=extra.get("threshold_bb") if final else None,
            threshold_label=step.threshold_label,
            prior_kind=pk,
            z_color_lim=z_color_lim,
            show_surface=step.show_surface,
            surface_grid=step.surface_grid,
            squish_u=step.squish_u,
            data_xlim=extra.get("data_xlim", step.data_xlim),
            data_ylim=extra.get("data_ylim", step.data_ylim),
            show_shadow_marker=bool(extra.get("show_shadow_marker", False)),
            shadow_marker_ws=extra.get("shadow_marker_ws"),
            shadow_marker_we=extra.get("shadow_marker_we"),
            show_shadow_threshold=bool(extra.get("show_shadow_threshold", False)),
            shadow_threshold_ws=extra.get("shadow_threshold_ws"),
            shadow_threshold_we=extra.get("shadow_threshold_we"),
            shadow_threshold_bb=extra.get("shadow_threshold_bb"),
            origin_guides=extra.get("origin_guides", step.origin_guides),
            zero_axis_guides_u=float(extra.get("zero_axis_guides_u", step.zero_axis_guides_u)),
        )
        cells[i][j] = _ch5_render_landscape_step(cell_step, frame_kwargs)
    return cells


def _ch5_render_landscape_grid(
    step: _LandscapeRenderStep,
    datasets: dict[str, dict],
    visible_keys: set[str],
    frame_kwargs: dict | None,
    *,
    per_key: dict[str, dict] | None = None,
) -> Any:
    """Composite one HQ landscape cell per visible dataset into a 2×2 grid."""
    cells = _ch5_render_landscape_grid_cells(
        step, datasets, visible_keys, frame_kwargs, per_key=per_key,
    )
    return ch5_composite_2x2_quadrants(cells)


def _ch5_landscape_grid_final_pack(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
) -> dict:
    """Final posterior meshes + per-cell render extras (all datasets complete)."""
    cfg = ch5_prior_landscape_config(hq=True) if config is None else config
    pk = "uniform"
    z_lim_final = ch5_belief_landscape_z_lim(pk, phase_u=1.0)
    z_color_lim = ch5_uniform_belief_z_lim()
    per_key: dict[str, dict] = {}
    for key in CH5_DATASET_KEYS:
        ds = datasets[key]
        study = np.asarray(ds["study"], dtype=np.float64)
        exam = np.asarray(ds["exam"], dtype=np.float64)
        y_arr = np.asarray(ds["y"], dtype=np.int64)
        mesh_n = ch5_posterior_w12_mesh_pack(
            study, exam, y_arr, prior_kind=pk, config=cfg, z_lim=z_lim_final,
        )
        per_key[key] = {
            "mesh_pack": mesh_n,
            "z_lim": z_lim_final,
            "z_ref": float(np.nanmax(mesh_n["Z"])),
            "study": study,
            "exam": exam,
            "y": y_arr,
            "final": True,
            "marker_ws": float(mesh_n["ws"]),
            "marker_we": float(mesh_n["we"]),
            "marker_z": float(mesh_n["marker_z"]),
            "threshold_ws": float(mesh_n["ws"]),
            "threshold_we": float(mesh_n["we"]),
            "threshold_bb": float(mesh_n["bb"]),
        }
    return {
        "config": cfg,
        "prior_kind": pk,
        "z_lim_final": z_lim_final,
        "z_color_lim": z_color_lim,
        "per_key": per_key,
    }


def _ch5_landscape_grid_base_step(
    *,
    ws: float,
    we: float,
    bb: float,
    mesh_prior: dict,
    z_lim_prior: tuple[float, float],
    z_ref_prior: float,
    elev: float,
    azim: float,
    pk: str,
    z_color_lim: tuple[float, float],
    squish_u: float = 0.0,
) -> _LandscapeRenderStep:
    empty_study = np.array([], dtype=np.float64)
    empty_exam = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)
    return _LandscapeRenderStep(
        ws=ws, we=we, bb=bb,
        elev=float(elev), azim=float(azim),
        mesh_pack=mesh_prior,
        z_lim=z_lim_prior,
        z_ref=z_ref_prior,
        study=empty_study,
        exam=empty_exam,
        y=empty_y,
        show_curves=False,
        marker=False,
        lrev=1.0,
        landscape_reveal_origin="lo_hi",
        knobs_zero=True,
        prior_kind=pk,
        z_color_lim=z_color_lim,
        show_surface=True,
        surface_grid=True,
        squish_u=float(squish_u),
    )


def _ch5_render_landscape_single(
    step: _LandscapeRenderStep,
    datasets: dict[str, dict],
    key: str,
    frame_kwargs: dict | None,
    per_key: dict[str, dict],
) -> Any:
    """One full duo frame for a single dataset (not embedded in a 2×2 composite)."""
    extra = dict(per_key.get(key, {}))
    mesh_pack = extra.get("mesh_pack", step.mesh_pack)
    z_lim = extra.get("z_lim", step.z_lim)
    z_ref = extra.get("z_ref", step.z_ref)
    study = extra.get("study", step.study)
    exam = extra.get("exam", step.exam)
    y_arr = extra.get("y", step.y)
    pk = step.prior_kind
    z_color_lim = step.z_color_lim or ch5_uniform_belief_z_lim()
    cell_step = _LandscapeRenderStep(
        ws=step.ws,
        we=step.we,
        bb=step.bb,
        elev=step.elev,
        azim=step.azim,
        mesh_pack=mesh_pack,
        z_lim=z_lim,
        z_ref=z_ref,
        curves=step.curves,
        study=study,
        exam=exam,
        y=y_arr,
        emp=step.emp,
        lrev=step.lrev,
        show_curves=step.show_curves,
        marker=True,
        landscape_reveal_origin=step.landscape_reveal_origin,
        knobs_zero=step.knobs_zero,
        marker_ws=extra.get("marker_ws"),
        marker_we=extra.get("marker_we"),
        marker_z=extra.get("marker_z"),
        here_annotation=True,
        here_label=step.here_label,
        show_threshold=True,
        threshold_ws=extra.get("threshold_ws"),
        threshold_we=extra.get("threshold_we"),
        threshold_bb=extra.get("threshold_bb"),
        threshold_label=step.threshold_label,
        prior_kind=pk,
        z_color_lim=z_color_lim,
        show_surface=step.show_surface,
        surface_grid=step.surface_grid,
        squish_u=step.squish_u,
    )
    return _ch5_render_landscape_step(cell_step, frame_kwargs)


def ch5_build_uniform_landscape_grid_zoom_frames(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    n_zoom: int | None = None,
    n_zoom_hold: int | None = None,
    n_orbit: int | None = None,
    n_focus_fade: int | None = None,
    dim_grey: float | None = None,
    dim_alpha: float | None = None,
) -> list:
    """
    Continue from ch5_47 end pose: grey inactive cells, zoom into each dataset,
    360° spin, zoom back out.
    """
    cfg = ch5_prior_landscape_config(hq=True) if config is None else config
    fk = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    final = _ch5_landscape_grid_final_pack(datasets, config=cfg)
    per_key = final["per_key"]
    pk = final["prior_kind"]
    z_color_lim = final["z_color_lim"]
    wz, ez, bz = CH5_KNOB_ZERO
    el_land = _ch5_hq_land_elev()
    az_base = float(_g("CH3_LIK_W12_CT_AZIM"))
    z_lim_prior = ch5_prior_w12_z_lim(pk, scope="prior")
    empty_study = np.array([], dtype=np.float64)
    empty_exam = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)

    n_z = int(CH5_HQ_GRID_N_ZOOM if n_zoom is None else n_zoom)
    n_hold = int(CH5_HQ_GRID_N_ZOOM_HOLD if n_zoom_hold is None else n_zoom_hold)
    n_spin = int(CH5_HQ_GRID_N_ZOOM_ORBIT if n_orbit is None else n_orbit)
    n_fade = int(CH5_HQ_GRID_N_FOCUS_FADE if n_focus_fade is None else n_focus_fade)
    orbit_deg = float(CH5_HQ_GRID_ORBIT_DEG)
    grey_w = float(CH5_HQ_GRID_FOCUS_DIM_GREY if dim_grey is None else dim_grey)
    alpha_min = float(CH5_HQ_GRID_FOCUS_DIM_ALPHA if dim_alpha is None else dim_alpha)
    focus_kw = dict(grey_weight=grey_w, alpha_min=alpha_min)

    frames: list = []
    with _landscape_render_context(cfg.dpi):
        mesh_prior = ch5_posterior_w12_mesh_pack(
            empty_study, empty_exam, empty_y, prior_kind=pk, config=cfg,
            z_lim=z_lim_prior,
        )
        z_ref_prior = float(np.nanmax(mesh_prior["Z"]))
        base_step = _ch5_landscape_grid_base_step(
            ws=wz, we=ez, bb=bz,
            mesh_prior=mesh_prior,
            z_lim_prior=z_lim_prior,
            z_ref_prior=z_ref_prior,
            elev=el_land,
            azim=az_base,
            pk=pk,
            z_color_lim=z_color_lim,
        )
        cells = _ch5_render_landscape_grid_cells(
            base_step, datasets, set(CH5_DATASET_KEYS), fk, per_key=per_key,
        )
        grid_full = ch5_composite_2x2_quadrants(cells)

        for key in CH5_DATASET_KEYS:
            row, col = _CH5_GRID_SLOTS[key]
            cell_ref = _ch5_render_landscape_single(
                base_step, datasets, key, fk, per_key,
            )
            for tv in np.linspace(0.0, 1.0, n_fade, endpoint=True):
                frames.append(ch5_composite_2x2_focus(
                    cells, key, prev_focus=None, transition_u=float(tv), **focus_kw,
                ))
                _ch5_maybe_gc(frames)
            grid_focus = ch5_composite_2x2_focus(cells, key, dim_u=1.0, **focus_kw)
            for _ in range(max(1, n_hold // 2)):
                frames.append(grid_focus.copy() if hasattr(grid_focus, "copy") else grid_focus)
                _ch5_maybe_gc(frames)
            for tv in np.linspace(0.0, 1.0, n_z, endpoint=True):
                frames.append(ch5_quadrant_zoom_frame(grid_focus, row, col, float(tv)))
                _ch5_maybe_gc(frames)
            for _ in range(max(1, n_hold)):
                frames.append(cell_ref.copy() if hasattr(cell_ref, "copy") else cell_ref)
            # endpoint=True so the last frame lands on az_base + 360° (full turn).
            for tv in np.linspace(0.0, 1.0, n_spin, endpoint=True):
                az = az_base + orbit_deg * float(tv)
                spin_step = _ch5_landscape_grid_base_step(
                    ws=wz, we=ez, bb=bz,
                    mesh_prior=mesh_prior,
                    z_lim_prior=z_lim_prior,
                    z_ref_prior=z_ref_prior,
                    elev=el_land,
                    azim=az,
                    pk=pk,
                    z_color_lim=z_color_lim,
                )
                frames.append(_ch5_render_landscape_single(
                    spin_step, datasets, key, fk, per_key,
                ))
                _ch5_maybe_gc(frames)
            for tv in np.linspace(1.0, 0.0, n_z, endpoint=True):
                frames.append(ch5_quadrant_zoom_frame(grid_full, row, col, float(tv)))
                _ch5_maybe_gc(frames)

    return frames


def ch5_build_uniform_landscape_grid_d4_d2_zoom_frames(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    n_open_hold: int | None = None,
    n_focus_hold: int | None = None,
    n_focus_fade: int | None = None,
    n_zoom: int | None = None,
    n_zoom_hold: int | None = None,
    n_orbit: int | None = None,
    dim_grey: float | None = None,
    dim_alpha: float | None = None,
) -> list:
    """
    From ch5_47/48 grid: emphasize D4 → add D2 → zoom/orbit D4 (ch5_48 style).
    """
    cfg = ch5_prior_landscape_config(hq=True) if config is None else config
    fk = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    final = _ch5_landscape_grid_final_pack(datasets, config=cfg)
    per_key = final["per_key"]
    pk = final["prior_kind"]
    z_color_lim = final["z_color_lim"]
    wz, ez, bz = CH5_KNOB_ZERO
    el_land = _ch5_hq_land_elev()
    az_base = float(_g("CH3_LIK_W12_CT_AZIM"))
    z_lim_prior = ch5_prior_w12_z_lim(pk, scope="prior")
    empty_study = np.array([], dtype=np.float64)
    empty_exam = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)

    n_open = int(CH5_HQ_GRID_N_FOCUS_OPEN if n_open_hold is None else n_open_hold)
    n_fhold = int(CH5_HQ_GRID_N_FOCUS_HOLD if n_focus_hold is None else n_focus_hold)
    n_fade = int(CH5_HQ_GRID_N_FOCUS_FADE if n_focus_fade is None else n_focus_fade)
    n_z = int(CH5_HQ_GRID_N_ZOOM if n_zoom is None else n_zoom)
    n_hold = int(CH5_HQ_GRID_N_ZOOM_HOLD if n_zoom_hold is None else n_zoom_hold)
    n_spin = int(CH5_HQ_GRID_N_ZOOM_ORBIT if n_orbit is None else n_orbit)
    orbit_deg = float(CH5_HQ_GRID_ORBIT_DEG)
    grey_w = float(CH5_HQ_GRID_FOCUS_DIM_GREY if dim_grey is None else dim_grey)
    alpha_min = float(CH5_HQ_GRID_FOCUS_DIM_ALPHA if dim_alpha is None else dim_alpha)
    focus_kw = dict(grey_weight=grey_w, alpha_min=alpha_min)

    d4_row, d4_col = _CH5_GRID_SLOTS["D4"]
    frames: list = []
    with _landscape_render_context(cfg.dpi):
        mesh_prior = ch5_posterior_w12_mesh_pack(
            empty_study, empty_exam, empty_y, prior_kind=pk, config=cfg,
            z_lim=z_lim_prior,
        )
        z_ref_prior = float(np.nanmax(mesh_prior["Z"]))
        base_step = _ch5_landscape_grid_base_step(
            ws=wz, we=ez, bb=bz,
            mesh_prior=mesh_prior,
            z_lim_prior=z_lim_prior,
            z_ref_prior=z_ref_prior,
            elev=el_land,
            azim=az_base,
            pk=pk,
            z_color_lim=z_color_lim,
        )
        cells = _ch5_render_landscape_grid_cells(
            base_step, datasets, set(CH5_DATASET_KEYS), fk, per_key=per_key,
        )
        grid_full = ch5_composite_2x2_quadrants(cells)
        cell_d4 = _ch5_render_landscape_single(
            base_step, datasets, "D4", fk, per_key,
        )

        for _ in range(max(1, n_open)):
            frames.append(grid_full.copy() if hasattr(grid_full, "copy") else grid_full)
            _ch5_maybe_gc(frames)

        for tv in np.linspace(0.0, 1.0, n_fade, endpoint=True):
            frames.append(ch5_composite_2x2_focus(
                cells, "D4", prev_focus=None, transition_u=float(tv), **focus_kw,
            ))
            _ch5_maybe_gc(frames)
        grid_d4 = ch5_composite_2x2_focus(cells, "D4", dim_u=1.0, **focus_kw)
        for _ in range(max(1, n_fhold // 2)):
            frames.append(grid_d4.copy() if hasattr(grid_d4, "copy") else grid_d4)
            _ch5_maybe_gc(frames)

        for tv in np.linspace(0.0, 1.0, n_fade, endpoint=True):
            frames.append(ch5_composite_2x2_focus(
                cells, focus_keys=frozenset({"D4"}),
                add_lit_key="D2", add_lit_u=float(tv), **focus_kw,
            ))
            _ch5_maybe_gc(frames)
        grid_d4_d2 = ch5_composite_2x2_focus(
            cells, focus_keys=frozenset({"D4", "D2"}), dim_u=1.0, **focus_kw,
        )
        for _ in range(max(1, n_fhold)):
            frames.append(grid_d4_d2.copy() if hasattr(grid_d4_d2, "copy") else grid_d4_d2)
            _ch5_maybe_gc(frames)

        for tv in np.linspace(0.0, 1.0, n_z, endpoint=True):
            frames.append(ch5_quadrant_zoom_frame(grid_d4_d2, d4_row, d4_col, float(tv)))
            _ch5_maybe_gc(frames)
        for _ in range(max(1, n_hold)):
            frames.append(cell_d4.copy() if hasattr(cell_d4, "copy") else cell_d4)
            _ch5_maybe_gc(frames)
        # endpoint=True so the last frame lands on az_base + 360° (full turn).
        for tv in np.linspace(0.0, 1.0, n_spin, endpoint=True):
            az = az_base + orbit_deg * float(tv)
            spin_step = _ch5_landscape_grid_base_step(
                ws=wz, we=ez, bb=bz,
                mesh_prior=mesh_prior,
                z_lim_prior=z_lim_prior,
                z_ref_prior=z_ref_prior,
                elev=el_land,
                azim=az,
                pk=pk,
                z_color_lim=z_color_lim,
            )
            frames.append(_ch5_render_landscape_single(
                spin_step, datasets, "D4", fk, per_key,
            ))
            _ch5_maybe_gc(frames)

    return frames


def _ch5_map_perturb_step(
    *,
    ws: float,
    we: float,
    bb: float,
    elev: float,
    azim: float,
    emp: str | None,
    extra: dict,
    pk: str,
    z_color_lim,
) -> _LandscapeRenderStep:
    """MAP pinned on 3D/2D; knob (ws, we) drives the moving shadow.

    ``emp`` engages a dial only while that knob is moving; ``None`` keeps
    MAP dial angles with equal/unset emphasis (camera rotation / holds).
    """
    mws = float(extra["marker_ws"])
    mwe = float(extra["marker_we"])
    mbb = float(extra["threshold_bb"])
    show_shadow = abs(float(ws) - mws) > 1e-5 or abs(float(we) - mwe) > 1e-5
    return _LandscapeRenderStep(
        ws=float(ws), we=float(we), bb=float(bb),
        elev=float(elev), azim=float(azim),
        mesh_pack=extra["mesh_pack"],
        z_lim=extra["z_lim"],
        z_ref=extra["z_ref"],
        curves=[],
        study=extra["study"],
        exam=extra["exam"],
        y=extra["y"],
        emp=None if emp is None else str(emp),
        lrev=1.0,
        show_curves=False,
        marker=True,
        marker_ws=mws,
        marker_we=mwe,
        marker_z=extra.get("marker_z"),
        here_annotation=True,
        here_label="most plausible line",
        show_threshold=True,
        threshold_ws=mws,
        threshold_we=mwe,
        threshold_bb=mbb,
        threshold_label="most plausible line",
        show_shadow_marker=show_shadow,
        shadow_marker_ws=float(ws),
        shadow_marker_we=float(we),
        show_shadow_threshold=show_shadow,
        shadow_threshold_ws=float(ws),
        shadow_threshold_we=float(we),
        shadow_threshold_bb=float(bb),
        knobs_zero=False,
        prior_kind=pk,
        z_color_lim=z_color_lim,
        show_surface=True,
        surface_grid=True,
    )


def ch5_build_posterior_map_perturb_frames(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    dataset_keys: tuple[str, ...] | None = None,
    n_rot: int | None = None,
    n_knob: int | None = None,
    n_hold: int | None = None,
    rot_deg: float | None = None,
    dw_el: float | None = None,
    dw_st: float | None = None,
) -> list:
    """
    Per dataset: MAP posterior + 90° CCW camera → w_EL shadow up → w_ST shadow
    down → return to MAP → camera back.
    """
    cfg = ch5_grid_landscape_config() if config is None else config
    fk = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    final = _ch5_landscape_grid_final_pack(datasets, config=cfg)
    per_key = final["per_key"]
    pk = final["prior_kind"]
    z_color_lim = final["z_color_lim"]
    el_ct = _ch5_hq_land_elev()
    az_ct = float(_g("CH3_LIK_W12_CT_AZIM"))
    lo, hi = float(CH5_VIEW_BOUNDS[0]), float(CH5_VIEW_BOUNDS[1])

    n_spin = int(CH5_MAP_PERTURB_N_ROT if n_rot is None else n_rot)
    n_k = int(CH5_MAP_PERTURB_N_KNOB if n_knob is None else n_knob)
    n_h = int(CH5_MAP_PERTURB_N_HOLD if n_hold is None else n_hold)
    rdeg = float(CH5_MAP_PERTURB_ROT_DEG if rot_deg is None else rot_deg)
    delt_el = float(CH5_MAP_PERTURB_DW_EL if dw_el is None else dw_el)
    delt_st = float(CH5_MAP_PERTURB_DW_ST if dw_st is None else dw_st)
    keys = tuple(CH5_DATASET_KEYS if dataset_keys is None else dataset_keys)

    smooth = _g("ch3_knob_smoothstep")
    lerp = _g("ch3_lerp")
    frames: list = []

    def _emit(step: _LandscapeRenderStep, *, hold: int = 1) -> None:
        img = _ch5_render_landscape_step(step, fk)
        for _ in range(max(1, int(hold))):
            frames.append(img.copy() if hasattr(img, "copy") else img)
        _ch5_maybe_gc(frames)

    with _landscape_render_context(cfg.dpi):
        for key in keys:
            extra = dict(per_key[key])
            mws = float(extra["marker_ws"])
            mwe = float(extra["marker_we"])
            mbb = float(extra["threshold_bb"])
            dwe = min(delt_el, max(0.15, hi - mwe - 0.08))
            dws = min(delt_st, max(0.15, mws - lo - 0.08))
            we_up = mwe + dwe
            ws_dn = mws - dws
            az_rot = az_ct + rdeg

            # Camera / holds: dials show MAP, none engaged.
            map_step = _ch5_map_perturb_step(
                ws=mws, we=mwe, bb=mbb, elev=el_ct, azim=az_ct, emp=None,
                extra=extra, pk=pk, z_color_lim=z_color_lim,
            )
            _emit(map_step, hold=n_h)

            for tv in np.linspace(0.0, 1.0, n_spin, endpoint=True):
                u = float(smooth(float(tv)))
                az = float(az_ct) + rdeg * u
                _emit(_ch5_map_perturb_step(
                    ws=mws, we=mwe, bb=mbb, elev=el_ct, azim=az, emp=None,
                    extra=extra, pk=pk, z_color_lim=z_color_lim,
                ))
            _emit(_ch5_map_perturb_step(
                ws=mws, we=mwe, bb=mbb, elev=el_ct, azim=az_rot, emp=None,
                extra=extra, pk=pk, z_color_lim=z_color_lim,
            ), hold=max(1, n_h // 2))

            # Engage only while a knob is moving.
            for tv in np.linspace(0.0, 1.0, n_k, endpoint=True):
                we_k = float(lerp(mwe, we_up, float(smooth(float(tv)))))
                _emit(_ch5_map_perturb_step(
                    ws=mws, we=we_k, bb=mbb, elev=el_ct, azim=az_rot, emp="el",
                    extra=extra, pk=pk, z_color_lim=z_color_lim,
                ))
            _emit(_ch5_map_perturb_step(
                ws=mws, we=we_up, bb=mbb, elev=el_ct, azim=az_rot, emp=None,
                extra=extra, pk=pk, z_color_lim=z_color_lim,
            ), hold=max(1, n_h // 2))
            for tv in np.linspace(0.0, 1.0, n_k, endpoint=True):
                we_k = float(lerp(we_up, mwe, float(smooth(float(tv)))))
                _emit(_ch5_map_perturb_step(
                    ws=mws, we=we_k, bb=mbb, elev=el_ct, azim=az_rot, emp="el",
                    extra=extra, pk=pk, z_color_lim=z_color_lim,
                ))

            for tv in np.linspace(0.0, 1.0, n_k, endpoint=True):
                ws_k = float(lerp(mws, ws_dn, float(smooth(float(tv)))))
                _emit(_ch5_map_perturb_step(
                    ws=ws_k, we=mwe, bb=mbb, elev=el_ct, azim=az_rot, emp="st",
                    extra=extra, pk=pk, z_color_lim=z_color_lim,
                ))
            _emit(_ch5_map_perturb_step(
                ws=ws_dn, we=mwe, bb=mbb, elev=el_ct, azim=az_rot, emp=None,
                extra=extra, pk=pk, z_color_lim=z_color_lim,
            ), hold=max(1, n_h // 2))
            for tv in np.linspace(0.0, 1.0, n_k, endpoint=True):
                ws_k = float(lerp(ws_dn, mws, float(smooth(float(tv)))))
                _emit(_ch5_map_perturb_step(
                    ws=ws_k, we=mwe, bb=mbb, elev=el_ct, azim=az_rot, emp="st",
                    extra=extra, pk=pk, z_color_lim=z_color_lim,
                ))

            _emit(_ch5_map_perturb_step(
                ws=mws, we=mwe, bb=mbb, elev=el_ct, azim=az_rot, emp=None,
                extra=extra, pk=pk, z_color_lim=z_color_lim,
            ), hold=n_h)

            for tv in np.linspace(0.0, 1.0, n_spin, endpoint=True):
                u = float(smooth(float(tv)))
                az = float(az_rot) - rdeg * u
                _emit(_ch5_map_perturb_step(
                    ws=mws, we=mwe, bb=mbb, elev=el_ct, azim=az, emp=None,
                    extra=extra, pk=pk, z_color_lim=z_color_lim,
                ))

    return frames


def _ch5_sigma_mesh_for_limits(xlim, ylim):
    """σ colormap mesh over the visible 2D panel."""
    ref_st, ref_el = _g("ST_KNOB"), _g("EL_KNOB")
    st = np.linspace(float(xlim[0]), float(xlim[1]), int(ref_st.shape[1]))
    el = np.linspace(float(ylim[0]), float(ylim[1]), int(ref_el.shape[0]))
    return np.meshgrid(st, el)


def _ch5_colormap_reveal_sigma_Z(
    ws: float,
    we: float,
    bb: float,
    sigma_stg,
    sigma_elg,
    reveal_u: float,
) -> np.ndarray:
    """Masked P(pass) colormap expanding outward from the decision boundary."""
    logits = _g("logits_plane")(float(ws), float(we), float(bb), sigma_stg, sigma_elg)
    z = _g("sigmoid")(logits)
    abs_log = np.abs(np.asarray(logits, dtype=float))
    dmax = float(np.nanmax(abs_log))
    u = float(_g("ch3_knob_smoothstep")(np.clip(float(reveal_u), 0.0, 1.0)))
    eps = 1e-6
    edge = eps + max(dmax - eps, 0.0) * u
    return np.where(abs_log <= edge, z, np.nan)


def _ch5_belief_z_at(mesh_pack: dict, ws: float, we: float) -> float:
    z_at = _g("_ch3_lik_w12_z_at")
    return float(z_at(mesh_pack["W1m"], mesh_pack["W2m"], mesh_pack["Z"], float(ws), float(we)))


def _ch5_d4_origin_guides(
    mesh_pack: dict,
    z_lim: tuple[float, float],
    map_z: float,
    *,
    el_tick_u: float = 0.0,
    st_tick_u: float = 0.0,
    el_floor_u: float = 0.0,
    st_floor_u: float = 0.0,
    belief_axis_u: float = 0.0,
    lines_alpha: float = 1.0,
) -> dict:
    return {
        "w1_lo": float(mesh_pack["w1_lo"]),
        "w1_hi": float(mesh_pack["w1_hi"]),
        "w2_lo": float(mesh_pack["w2_lo"]),
        "w2_hi": float(mesh_pack["w2_hi"]),
        "z_floor": float(z_lim[0]),
        "map_z": float(map_z),
        "el_tick_u": float(el_tick_u),
        "st_tick_u": float(st_tick_u),
        "el_floor_u": float(el_floor_u),
        "st_floor_u": float(st_floor_u),
        "belief_axis_u": float(belief_axis_u),
        "lines_alpha": float(lines_alpha),
    }


def _ch5_active_knob_scale() -> float:
    return float(_g("CH3_KNOB_ACTIVE_SCALE"))


def _ch5_dual_knob_scales() -> list[float]:
    sc = _ch5_active_knob_scale()
    return [sc, sc, 1.0]


def _ch5_d4_origin_tutorial_step(
    *,
    ws: float,
    we: float,
    bb: float,
    elev: float,
    azim: float,
    emp: str | None,
    extra: dict,
    pk: str,
    z_color_lim,
    here_annotation: bool = True,
    show_threshold: bool = True,
    cmap_reveal_u: float = 0.0,
    origin_guides: dict | None = None,
    knobs_zero: bool = False,
    knob_scales: list | None = None,
    pin_map: bool = False,
) -> _LandscapeRenderStep:
    """D4 tutorial frame.

    Pinned MAP frames use the same knobs_zero + marker path as ch5_51's zoomed
    D4 cell so the belief surface colors match that clip.
    """
    mesh_pack = extra["mesh_pack"]
    mws = float(extra["marker_ws"])
    mwe = float(extra["marker_we"])
    mbb = float(extra["threshold_bb"])
    map_z = float(extra["marker_z"])
    line_ws = float(ws)
    line_we = float(we)
    line_bb = float(bb)
    if pin_map or knobs_zero:
        line_ws, line_we, line_bb = mws, mwe, mbb
    # Match ch5_51: always use the packed MAP marker_z on pinned frames.
    if pin_map or knobs_zero:
        mz = map_z
        mark_ws, mark_we = mws, mwe
    else:
        mz = _ch5_belief_z_at(mesh_pack, ws, we)
        mark_ws, mark_we = float(ws), float(we)
    sigma_z = None
    show_cmap = False
    if float(cmap_reveal_u) > 1e-5:
        xl, yl = ch5_plot_limits("D4")
        stg, elg = _ch5_sigma_mesh_for_limits(xl, yl)
        sigma_z = _ch5_colormap_reveal_sigma_Z(line_ws, line_we, line_bb, stg, elg, cmap_reveal_u)
        show_cmap = True
    wz, ez, bz = CH5_KNOB_ZERO
    return _LandscapeRenderStep(
        ws=float(wz if knobs_zero else ws),
        we=float(ez if knobs_zero else we),
        bb=float(bz if knobs_zero else bb),
        elev=float(elev), azim=float(azim),
        mesh_pack=mesh_pack,
        z_lim=extra["z_lim"],
        z_ref=extra["z_ref"],
        curves=[],
        study=extra["study"],
        exam=extra["exam"],
        y=extra["y"],
        emp="st" if emp is None else str(emp),
        lrev=1.0,
        show_curves=False,
        marker=True,
        marker_ws=mark_ws,
        marker_we=mark_we,
        marker_z=mz,
        here_annotation=bool(here_annotation),
        here_label="most plausible line",
        show_threshold=bool(show_threshold),
        threshold_ws=line_ws,
        threshold_we=line_we,
        threshold_bb=line_bb,
        threshold_label="most plausible line",
        knobs_zero=bool(knobs_zero),
        knob_scales=knob_scales,
        prior_kind=pk,
        z_color_lim=z_color_lim,
        show_surface=True,
        surface_grid=True,
        origin_guides=origin_guides,
        sigma_Z=sigma_z,
        show_colormap_panel=show_cmap,
    )


def ch5_build_d4_origin_map_tutorial_frames(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    n_guide: int | None = None,
    n_fade: int | None = None,
    n_cmap: int | None = None,
    n_knob: int | None = None,
    n_hold: int | None = None,
    dw_el: float | None = None,
    dw_st: float | None = None,
) -> list:
    """
    D4 belief landscape: emphasize w_EL=0 / w_ST=0 guides → colormap reveal →
    coupled MAP perturbations → restore opening pose.
    """
    cfg = ch5_grid_landscape_config() if config is None else config
    fk = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    final = _ch5_landscape_grid_final_pack(datasets, config=cfg)
    per_key = final["per_key"]
    pk = final["prior_kind"]
    z_color_lim = final["z_color_lim"]
    extra = dict(per_key["D4"])
    el_ct = _ch5_hq_land_elev()
    az_ct = float(_g("CH3_LIK_W12_CT_AZIM"))
    lo, hi = float(CH5_VIEW_BOUNDS[0]), float(CH5_VIEW_BOUNDS[1])

    n_g = int(CH5_D4_ORIGIN_N_GUIDE if n_guide is None else n_guide)
    n_f = int(CH5_D4_ORIGIN_N_FADE if n_fade is None else n_fade)
    n_c = int(CH5_D4_ORIGIN_N_CMAP if n_cmap is None else n_cmap)
    n_k = int(CH5_D4_ORIGIN_N_KNOB if n_knob is None else n_knob)
    n_h = int(CH5_D4_ORIGIN_N_HOLD if n_hold is None else n_hold)
    delt_el = float(CH5_D4_ORIGIN_DW_EL if dw_el is None else dw_el)
    delt_st = float(CH5_D4_ORIGIN_DW_ST if dw_st is None else dw_st)

    mws = float(extra["marker_ws"])
    mwe = float(extra["marker_we"])
    mbb = float(extra["threshold_bb"])
    map_z = float(extra["marker_z"])
    mesh_pack = extra["mesh_pack"]
    z_lim = extra["z_lim"]

    dwe = min(delt_el, max(0.15, hi - mwe - 0.08))
    dws = min(delt_st, max(0.15, mws - lo - 0.08))
    we_up, we_dn = mwe + dwe, mwe - dwe
    ws_up, ws_dn = mws + dws, mws - dws

    smooth = _g("ch3_knob_smoothstep")
    lerp = _g("ch3_lerp")
    frames: list = []

    def _emit(step: _LandscapeRenderStep, *, hold: int = 1) -> None:
        img = _ch5_render_landscape_step(step, fk)
        for _ in range(max(1, int(hold))):
            frames.append(img.copy() if hasattr(img, "copy") else img)
        _ch5_maybe_gc(frames)

    def _pinned_step(**kw) -> _LandscapeRenderStep:
        """Match ch5_47+ pose: unset knobs, MAP marker/threshold pinned."""
        return _ch5_d4_origin_tutorial_step(
            ws=mws, we=mwe, bb=mbb,
            elev=el_ct, azim=az_ct, emp=None,
            extra=extra, pk=pk, z_color_lim=z_color_lim,
            knobs_zero=True, pin_map=True, **kw,
        )

    def _guides_full(*, lines_alpha: float = 1.0) -> dict:
        return _ch5_d4_origin_guides(
            mesh_pack, z_lim, map_z,
            el_tick_u=1.0, st_tick_u=1.0,
            el_floor_u=1.0, st_floor_u=1.0,
            belief_axis_u=1.0, lines_alpha=lines_alpha,
        )

    def _knob_sweep(
        ws0: float, ws1: float,
        we0: float, we1: float,
        *,
        emp: str,
        cmap_u: float = 1.0,
        annot: bool = False,
        knob_scales: list | None = None,
    ) -> None:
        for tv in np.linspace(0.0, 1.0, n_k, endpoint=True):
            u = float(smooth(float(tv)))
            ws_k = float(lerp(ws0, ws1, u))
            we_k = float(lerp(we0, we1, u))
            _emit(_ch5_d4_origin_tutorial_step(
                ws=ws_k, we=we_k, bb=mbb, elev=el_ct, azim=az_ct, emp=emp,
                extra=extra, pk=pk, z_color_lim=z_color_lim,
                here_annotation=annot, cmap_reveal_u=cmap_u,
                knob_scales=knob_scales,
            ))
        _emit(_ch5_d4_origin_tutorial_step(
            ws=ws1, we=we1, bb=mbb, elev=el_ct, azim=az_ct, emp=emp,
            extra=extra, pk=pk, z_color_lim=z_color_lim,
            here_annotation=annot, cmap_reveal_u=cmap_u,
            knob_scales=knob_scales,
        ), hold=max(1, n_h // 2))
        for tv in np.linspace(0.0, 1.0, n_k, endpoint=True):
            u = float(smooth(float(tv)))
            ws_k = float(lerp(ws1, ws0, u))
            we_k = float(lerp(we1, we0, u))
            _emit(_ch5_d4_origin_tutorial_step(
                ws=ws_k, we=we_k, bb=mbb, elev=el_ct, azim=az_ct, emp=emp,
                extra=extra, pk=pk, z_color_lim=z_color_lim,
                here_annotation=annot, cmap_reveal_u=cmap_u,
                knob_scales=knob_scales,
            ))

    dual_scales = _ch5_dual_knob_scales()

    with _landscape_render_context(cfg.dpi):
        _emit(_pinned_step(here_annotation=True), hold=n_h)

        for tv in np.linspace(0.0, 1.0, n_g, endpoint=True):
            g_el_tick = float(smooth(float(tv)))
            guides = _ch5_d4_origin_guides(
                mesh_pack, z_lim, map_z, el_tick_u=g_el_tick,
            )
            _emit(_pinned_step(here_annotation=True, origin_guides=guides))

        for tv in np.linspace(0.0, 1.0, n_g, endpoint=True):
            g_el_floor = float(smooth(float(tv)))
            guides = _ch5_d4_origin_guides(
                mesh_pack, z_lim, map_z, el_tick_u=1.0, el_floor_u=g_el_floor,
            )
            _emit(_pinned_step(here_annotation=True, origin_guides=guides))

        for tv in np.linspace(0.0, 1.0, n_g, endpoint=True):
            g_st_tick = float(smooth(float(tv)))
            guides = _ch5_d4_origin_guides(
                mesh_pack, z_lim, map_z,
                el_tick_u=1.0, el_floor_u=1.0, st_tick_u=g_st_tick,
            )
            _emit(_pinned_step(here_annotation=True, origin_guides=guides))

        for tv in np.linspace(0.0, 1.0, n_g, endpoint=True):
            g_st_floor = float(smooth(float(tv)))
            guides = _ch5_d4_origin_guides(
                mesh_pack, z_lim, map_z,
                el_tick_u=1.0, el_floor_u=1.0,
                st_tick_u=1.0, st_floor_u=g_st_floor,
            )
            _emit(_pinned_step(here_annotation=True, origin_guides=guides))

        for tv in np.linspace(0.0, 1.0, n_g, endpoint=True):
            g_belief = float(smooth(float(tv)))
            guides = _ch5_d4_origin_guides(
                mesh_pack, z_lim, map_z,
                el_tick_u=1.0, el_floor_u=1.0,
                st_tick_u=1.0, st_floor_u=1.0,
                belief_axis_u=g_belief,
            )
            _emit(_pinned_step(here_annotation=True, origin_guides=guides))

        guides_full = _guides_full(lines_alpha=1.0)
        _emit(_pinned_step(here_annotation=True, origin_guides=guides_full), hold=max(1, n_h // 2))

        for tv in np.linspace(1.0, 0.0, n_f, endpoint=True):
            al = float(smooth(float(tv)))
            guides = _guides_full(lines_alpha=al)
            _emit(_pinned_step(here_annotation=True, origin_guides=guides))

        for tv in np.linspace(0.0, 1.0, n_c, endpoint=True):
            cu = float(smooth(float(tv)))
            _emit(_pinned_step(here_annotation=True, cmap_reveal_u=cu))

        _emit(_pinned_step(here_annotation=False, cmap_reveal_u=1.0), hold=max(1, n_h // 2))

        _knob_sweep(mws, ws_up, mwe, mwe, emp="st", cmap_u=1.0, annot=False)
        _knob_sweep(mws, ws_dn, mwe, mwe, emp="st", cmap_u=1.0, annot=False)
        _knob_sweep(mws, mws, mwe, we_up, emp="el", cmap_u=1.0, annot=False)
        _knob_sweep(mws, mws, mwe, we_dn, emp="el", cmap_u=1.0, annot=False)
        _knob_sweep(
            mws, ws_up, mwe, we_dn, emp="both", cmap_u=1.0, annot=False,
            knob_scales=dual_scales,
        )
        _knob_sweep(
            mws, ws_dn, mwe, we_up, emp="both", cmap_u=1.0, annot=False,
            knob_scales=dual_scales,
        )

        for tv in np.linspace(1.0, 0.0, n_c, endpoint=True):
            cu = float(smooth(float(tv)))
            _emit(_pinned_step(here_annotation=False, cmap_reveal_u=cu))

        _emit(_pinned_step(here_annotation=True), hold=n_h)

    return frames


def _ch5_lerp_limits(lo0, hi0, lo1, hi1, u: float) -> tuple[float, float]:
    return (
        float(lo0 + (lo1 - lo0) * u),
        float(hi0 + (hi1 - hi0) * u),
    )


def ch5_build_grid_2d_zoom_shadow_orbit_frames(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    n_zoom: int | None = None,
    n_hold: int | None = None,
    n_orbit: int | None = None,
    n_guide: int | None = None,
    n_cam: int | None = None,
    n_d4_swing: int | None = None,
    target_lim: tuple[float, float] | None = None,
    camera_pan: bool = False,
) -> list:
    """
    2×2 posterior grid: fade in ST/EL=0 crosses → zoom each 2D panel out to
    ±target (3D fixed) → optional tilt toward top-down → D4 shadow swings out
    (+w_ST, −w_EL) → all cells orbit one turn together → D4 swings back →
    optional cameras return to landscape elev while 2D panels zoom back in.

    Serial path: pack → FrameSpec iterator → render (same choreography as parallel).
    """
    pack = ch5_build_grid_2d_zoom_shadow_orbit_pack(
        datasets, config=config, frame_kwargs=frame_kwargs,
    )
    frames: list = []
    with _landscape_render_context(int(pack["config"].dpi)):
        for spec in ch5_iter_grid_2d_zoom_shadow_orbit_specs(
            pack,
            n_zoom=n_zoom,
            n_hold=n_hold,
            n_orbit=n_orbit,
            n_guide=n_guide,
            n_cam=n_cam,
            n_d4_swing=n_d4_swing,
            target_lim=target_lim,
            camera_pan=bool(camera_pan),
            hold_tail=0,
            clip_id="ch5_54" if not camera_pan else "ch5_56",
        ):
            frames.append(ch5_render_landscape_grid_from_spec(pack, spec))
            _ch5_maybe_gc(frames)
    return frames


def ch5_iter_grid_map_labeled_rotate90_specs(
    pack: dict,
    *,
    n_rot: int | None = None,
    n_hold: int | None = None,
    rot_deg: float | None = None,
    hold_tail: int = 0,
    clip_id: str = "ch5_57",
) -> Iterator[dict]:
    """Yield labeled 2×2 belief-grid specs that spin +90° CCW (no images)."""
    pk = pack["prior_kind"]
    z_lim_prior = pack["z_lim_prior"]
    z_color_lim = pack["z_color_lim"]
    meshes = pack["meshes"]
    wz, ez, bz = CH5_KNOB_ZERO
    el_land = _ch5_hq_land_elev()
    az_base = float(_g("CH3_LIK_W12_CT_AZIM"))
    mesh_prior_key = "__mesh_prior__"
    z_ref_prior = float(np.nanmax(meshes[mesh_prior_key]["Z"]))

    n_r = int(CH5_GRID_MAP_ROT_N if n_rot is None else n_rot)
    n_h = int(CH5_GRID_MAP_ROT_N_HOLD if n_hold is None else n_hold)
    deg = float(CH5_GRID_MAP_ROT_DEG if rot_deg is None else rot_deg)
    smooth = _g("ch3_knob_smoothstep")
    last_spec: dict | None = None

    per_key = {
        key: {
            "mesh_key": f"__final_mesh_{key}__",
            "final": True,
            "marker": True,
            "threshold": True,
            "here_annotation": True,
            "data_xlim": tuple(CH5_STANDARD_XLIM),
            "data_ylim": tuple(CH5_STANDARD_YLIM),
        }
        for key in CH5_DATASET_KEYS
    }

    def _make_spec(azim: float) -> dict:
        step = _landscape_step_spec(
            ws=wz, we=ez, bb=bz,
            elev=float(el_land), azim=float(azim),
            mesh_key=mesh_prior_key, z_lim=z_lim_prior, z_ref=z_ref_prior,
            knobs_zero=True, prior_kind=pk, z_color_lim=z_color_lim,
            show_surface=True, surface_grid=True, lrev=1.0,
            landscape_reveal_origin="lo_hi",
        )
        return {
            "clip_id": clip_id,
            "kind": "landscape_grid",
            "step": step,
            "visible_keys": tuple(CH5_DATASET_KEYS),
            "per_key": per_key,
        }

    def _emit(azim: float, *, hold: int = 1):
        nonlocal last_spec
        spec = _make_spec(azim)
        last_spec = spec
        for _ in range(max(1, int(hold))):
            yield dict(spec)

    # Opening hold with MAP point + most-plausible-line labels (as in 56).
    yield from _emit(az_base, hold=n_h)

    # Camera: all four belief plots rotate 90° counter-clockwise together.
    for tv in np.linspace(0.0, 1.0, n_r, endpoint=True):
        u = float(smooth(float(tv)))
        yield from _emit(float(az_base) + deg * u)

    yield from _emit(float(az_base) + deg, hold=n_h)

    if last_spec is not None:
        for _ in range(int(hold_tail)):
            yield dict(last_spec)


def ch5_build_grid_map_labeled_rotate90_frames(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    n_rot: int | None = None,
    n_hold: int | None = None,
    rot_deg: float | None = None,
) -> list:
    """
    2×2 posterior belief grid with MAP marker + most-plausible-line labels,
    then a shared 90° counter-clockwise camera spin on all cells.
    """
    pack = ch5_build_grid_2d_zoom_shadow_orbit_pack(
        datasets, config=config, frame_kwargs=frame_kwargs,
    )
    frames: list = []
    with _landscape_render_context(int(pack["config"].dpi)):
        for spec in ch5_iter_grid_map_labeled_rotate90_specs(
            pack,
            n_rot=n_rot,
            n_hold=n_hold,
            rot_deg=rot_deg,
            hold_tail=0,
            clip_id="ch5_57",
        ):
            frames.append(ch5_render_landscape_grid_from_spec(pack, spec))
            _ch5_maybe_gc(frames)
    return frames


def _ch5_stem_diag_rank(ws, we, *, w1_lo, w1_hi, w2_lo, w2_hi, origin: str) -> np.ndarray:
    """Diagonal progress in [0, 1] matching ``ch3_lik_w12_facecolors_diag``."""
    u1 = (np.asarray(ws, dtype=float) - float(w1_lo)) / max(float(w1_hi) - float(w1_lo), 1e-9)
    if str(origin) == "lo_hi":
        u2 = (float(w2_hi) - np.asarray(we, dtype=float)) / max(float(w2_hi) - float(w2_lo), 1e-9)
    else:
        u2 = (np.asarray(we, dtype=float) - float(w2_lo)) / max(float(w2_hi) - float(w2_lo), 1e-9)
    return 0.5 * (u1 + u2)


def ch5_build_belief_stem_surface_frames(
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    stem_stride: int | None = None,
    n_pillars: int | None = None,
    n_grow_per: int | None = None,
    n_grow: int | None = None,
    n_tighten: int | None = None,
    n_hold: int | None = None,
    n_morph: int | None = None,
    prior_kind: str = "uniform",
) -> list:
    """
    Fixed end-camera *prior* construction (empty 2D panel):

    1. Full-plane pillars rise one at a time (top = belief surface RGBA).
    2. Smooth prior surface wipes in while full-width pillars fade.
    """
    del stem_stride  # density is controlled by the 0.25 plane grid
    cfg = ch5_prior_landscape_config(hq=True) if config is None else config
    fk = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    pk = str(prior_kind).lower()
    z_lim = ch5_prior_w12_z_lim(pk, scope="prior")
    z_color_lim = ch5_uniform_belief_z_lim() if pk == "uniform" else z_lim
    # Prior only — no data on the 2D panel.
    study = np.array([], dtype=np.float64)
    exam = np.array([], dtype=np.float64)
    y_arr = np.array([], dtype=np.int64)
    mesh = ch5_posterior_w12_mesh_pack(
        study, exam, y_arr, prior_kind=pk, config=cfg, z_lim=z_lim,
    )
    el = _ch5_hq_land_elev()
    az = float(_g("CH3_LIK_W12_CT_AZIM"))
    wz, ez, bz = CH5_KNOB_ZERO
    origin = str(CH5_STEM_SURF_REVEAL_ORIGIN)
    n_gp = int(CH5_STEM_SURF_N_GROW_PER if n_grow_per is None else n_grow_per)
    if n_grow is not None and n_grow_per is None:
        n_gp = int(n_grow)
    n_h = int(CH5_STEM_SURF_N_HOLD if n_hold is None else n_hold)
    n_m = int(CH5_STEM_SURF_N_MORPH if n_morph is None else n_morph)
    del n_tighten  # no tighten phase
    hist_w = float(CH5_STEM_SURF_HIST_WIDTH)
    line_cut = float(CH5_STEM_SURF_LINE_WIDTH_FRAC)
    smooth = _g("ch3_knob_smoothstep")

    w1_lo, w1_hi = float(mesh["w1_lo"]), float(mesh["w1_hi"])
    w2_lo, w2_hi = float(mesh["w2_lo"]), float(mesh["w2_hi"])
    W1m = np.asarray(mesh["W1m"], dtype=float)
    W2m = np.asarray(mesh["W2m"], dtype=float)
    tip_z_full = np.asarray(mesh["Z"], dtype=float)
    sp = float(CH5_SURFACE_GRID_SPACING)
    # Tile the full (w_ST, w_EL) plane at grid spacing (span 6 → 24×24 = 576).
    n_side1 = max(1, int(round((w1_hi - w1_lo) / sp)))
    n_side2 = max(1, int(round((w2_hi - w2_lo) / sp)))
    centers1 = w1_lo + sp * (0.5 + np.arange(n_side1, dtype=float))
    centers2 = w2_lo + sp * (0.5 + np.arange(n_side2, dtype=float))
    CW1, CW2 = np.meshgrid(centers1, centers2, indexing="ij")
    SW1 = CW1.ravel()
    SW2 = CW2.ravel()
    n_grid = int(SW1.size)
    n_p = n_grid if n_pillars is None else int(np.clip(int(n_pillars), 1, n_grid))
    z_at = _g("_ch3_lik_w12_z_at")
    tip_z = np.asarray(
        [float(z_at(W1m, W2m, tip_z_full, float(SW1[i]), float(SW2[i]))) for i in range(n_grid)],
        dtype=float,
    )
    # Exact same RGBA as the finished belief surface (shared colormap + alpha).
    fc_surf, _ = _ch5_uniform_surface_facecolors(
        mesh,
        z_lim=z_lim,
        z_color_lim=z_color_lim,
        lrev=1.0,
        origin=origin,
        surface_alpha=float(CH5_BELIEF_SURFACE_ALPHA),
    )
    fc_flat = np.asarray(fc_surf, dtype=float).reshape(-1, 4)
    W1f = W1m.ravel()
    W2f = W2m.ravel()
    tip_rgba = np.empty((n_grid, 4), dtype=float)
    for i in range(n_grid):
        k = int(np.argmin((W1f - float(SW1[i])) ** 2 + (W2f - float(SW2[i])) ** 2))
        tip_rgba[i] = fc_flat[k]
    # Rise one-at-a-time along the diagonal cascade order.
    ranks = _ch5_stem_diag_rank(
        SW1, SW2,
        w1_lo=w1_lo, w1_hi=w1_hi, w2_lo=w2_lo, w2_hi=w2_hi, origin=origin,
    )
    order = np.argsort(ranks, kind="mergesort")[:n_p]
    SW1 = SW1[order]
    SW2 = SW2[order]
    tip_z = tip_z[order]
    tip_rgba = tip_rgba[order]
    dx_cell = sp
    dy_cell = sp
    frames: list = []

    def _stems_payload(
        heights: np.ndarray,
        *,
        width_frac: float,
        line_alpha: float,
        point_alpha_scale: float,
        tip_mask: np.ndarray | None = None,
        draw_tips: bool = True,
        bar_alpha: float | None = None,
        active: np.ndarray | None = None,
    ) -> dict:
        idx = np.arange(int(tip_z.size)) if active is None else np.asarray(active, dtype=int)
        rgba = np.array(tip_rgba[idx], copy=True)
        side_a = float(CH5_STEM_SURF_SIDE_ALPHA)
        if tip_mask is not None:
            m = np.asarray(tip_mask, dtype=float)
            if rgba.ndim == 2 and rgba.shape[1] >= 4:
                rgba[:, 3] = rgba[:, 3] * m[idx]
            side_a *= float(np.nanmean(m[idx])) if m.size else 1.0
        return {
            "ws": SW1[idx],
            "we": SW2[idx],
            "z": np.asarray(heights, dtype=float)[idx],
            "dx": float(dx_cell),
            "dy": float(dy_cell),
            "width_frac": float(width_frac),
            "line_width_frac": float(line_cut),
            "tip_rgba": rgba,
            # tip_rgba is the finished surface RGBA — do not rescale in the drawer.
            "bar_alpha": 1.0,
            "preserve_tip_alpha": True,
            "side_color": str(CH5_STEM_SURF_SIDE_COLOR),
            "side_alpha": float(side_a),
            "side_edge_color": str(CH5_STEM_SURF_SIDE_EDGE_COLOR),
            "side_edge_width": float(CH5_STEM_SURF_SIDE_EDGE_WIDTH),
            "line_color": str(CH5_STEM_SURF_LINE_COLOR),
            "line_alpha": float(line_alpha),
            "linewidth": float(CH5_STEM_SURF_LINEWIDTH),
            "point_size": float(CH5_STEM_SURF_POINT_SIZE),
            "point_alpha_scale": float(point_alpha_scale),
            "draw_tips": bool(draw_tips),
        }

    def _emit(
        *,
        heights: np.ndarray | None,
        width_frac: float,
        stem_line_alpha: float,
        stem_point_scale: float,
        surface_lrev: float,
        show_surface: bool,
        tip_mask: np.ndarray | None = None,
        draw_tips: bool = True,
        bar_alpha: float | None = None,
        active: np.ndarray | None = None,
        hold: int = 1,
    ) -> None:
        stems = None
        if heights is not None and (
            float(width_frac) > 1e-4 or stem_line_alpha > 1e-4 or stem_point_scale > 1e-4
        ):
            stems = _stems_payload(
                heights,
                width_frac=width_frac,
                line_alpha=stem_line_alpha,
                point_alpha_scale=stem_point_scale,
                tip_mask=tip_mask,
                draw_tips=draw_tips,
                bar_alpha=bar_alpha,
                active=active,
            )
        img = _ch5_frame_lik_w12_belief(
            study, exam, y_arr,
            float(wz), float(ez), float(bz),
            mesh_pack=mesh,
            z_lim=z_lim,
            prior_kind=pk,
            z_color_lim=z_color_lim,
            elev=el,
            azim=az,
            landscape_reveal=float(np.clip(surface_lrev, 0.0, 1.0)),
            landscape_reveal_origin=origin,
            show_curves=False,
            marker=False,
            show_threshold=False,
            show_surface=bool(show_surface) and float(surface_lrev) > 1e-6,
            surface_grid=bool(show_surface) and float(surface_lrev) > 1e-6 and pk == "uniform",
            belief_stems=stems,
            frame_kwargs=fk,
            **dict(CH5_KNOBS_UNSET_FRAME_KW),
        )
        for _ in range(max(1, int(hold))):
            frames.append(img.copy() if hasattr(img, "copy") else img)
        _ch5_maybe_gc(frames)

    with _landscape_render_context(cfg.dpi):
        ones = np.ones_like(tip_z)
        zeros = np.zeros_like(tip_z)
        all_idx = np.arange(n_p)
        # Opening: empty axes (no pillars yet).
        _emit(
            heights=None,
            width_frac=0.0,
            stem_line_alpha=0.0, stem_point_scale=0.0,
            surface_lrev=0.0, show_surface=False, tip_mask=None, draw_tips=False,
            hold=max(1, n_h // 2),
        )
        # 1) Strict sequential rise — only one pillar mid-flight at a time.
        for k in range(n_p):
            active = all_idx[: k + 1]
            for tv in np.linspace(0.0, 1.0, n_gp, endpoint=True):
                u = float(smooth(float(tv)))
                heights = tip_z.copy()
                heights[k + 1:] = 0.0
                heights[k] = float(tip_z[k]) * u
                _emit(
                    heights=heights,
                    width_frac=hist_w,
                    stem_line_alpha=0.0,
                    stem_point_scale=0.0,
                    surface_lrev=0.0, show_surface=False, tip_mask=ones, draw_tips=False,
                    active=active,
                )
        full_h = tip_z.copy()
        _emit(
            heights=full_h,
            width_frac=hist_w,
            stem_line_alpha=0.0,
            stem_point_scale=0.0,
            surface_lrev=0.0, show_surface=False, tip_mask=ones, draw_tips=False,
            active=all_idx, hold=n_h,
        )
        # 2) Smooth surface wipes in; full-width pillars fade (no shrink).
        for tv in np.linspace(0.0, 1.0, n_m, endpoint=True):
            u = float(smooth(float(tv)))
            fade = 1.0 - u
            _emit(
                heights=full_h,
                width_frac=hist_w,
                stem_line_alpha=0.0,
                stem_point_scale=0.0,
                surface_lrev=u, show_surface=True,
                tip_mask=ones * fade, draw_tips=False,
                active=all_idx,
            )
        _emit(
            heights=None,
            width_frac=0.0,
            stem_line_alpha=0.0, stem_point_scale=0.0,
            surface_lrev=1.0, show_surface=True, tip_mask=None, hold=n_h,
        )
    return frames


def ch5_build_uniform_landscape_squish_frames(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    n_squish: int | None = None,
) -> list:
    """Compress all four belief landscapes onto the (w_ST, w_EL) floor (z → 0)."""
    cfg = ch5_prior_landscape_config(hq=True) if config is None else config
    fk = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    final = _ch5_landscape_grid_final_pack(datasets, config=cfg)
    per_key = final["per_key"]
    pk = final["prior_kind"]
    z_color_lim = final["z_color_lim"]
    wz, ez, bz = CH5_KNOB_ZERO
    el_land = _ch5_hq_land_elev()
    az_land = float(_g("CH3_LIK_W12_CT_AZIM"))
    z_lim_prior = ch5_prior_w12_z_lim(pk, scope="prior")
    empty_study = np.array([], dtype=np.float64)
    empty_exam = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)
    n_sq = int(CH5_HQ_GRID_N_SQUISH if n_squish is None else n_squish)

    frames: list = []
    with _landscape_render_context(cfg.dpi):
        mesh_prior = ch5_posterior_w12_mesh_pack(
            empty_study, empty_exam, empty_y, prior_kind=pk, config=cfg,
            z_lim=z_lim_prior,
        )
        z_ref_prior = float(np.nanmax(mesh_prior["Z"]))
        for tv in np.linspace(0.0, 1.0, n_sq, endpoint=True):
            su = float(_g("ch3_knob_smoothstep")(float(tv)))
            step = _ch5_landscape_grid_base_step(
                ws=wz, we=ez, bb=bz,
                mesh_prior=mesh_prior,
                z_lim_prior=z_lim_prior,
                z_ref_prior=z_ref_prior,
                elev=el_land,
                azim=az_land,
                pk=pk,
                z_color_lim=z_color_lim,
                squish_u=su,
            )
            frames.append(_ch5_render_landscape_grid(
                step, datasets, set(CH5_DATASET_KEYS), fk, per_key=per_key,
            ))
            _ch5_maybe_gc(frames)
    return frames


def ch5_build_uniform_landscape_grid_focus_frames(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    n_open_hold: int | None = None,
    n_focus_hold: int | None = None,
    n_focus_fade: int | None = None,
    dim_grey: float | None = None,
    dim_alpha: float | None = None,
) -> list:
    """
    Continue from ch5_47 end: cycle D1–D4, greying/fading inactive quadrants.
    """
    cfg = ch5_prior_landscape_config(hq=True) if config is None else config
    fk = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    final = _ch5_landscape_grid_final_pack(datasets, config=cfg)
    per_key = final["per_key"]
    pk = final["prior_kind"]
    z_color_lim = final["z_color_lim"]
    wz, ez, bz = CH5_KNOB_ZERO
    el_land = _ch5_hq_land_elev()
    az_land = float(_g("CH3_LIK_W12_CT_AZIM"))
    z_lim_prior = ch5_prior_w12_z_lim(pk, scope="prior")
    empty_study = np.array([], dtype=np.float64)
    empty_exam = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)

    n_open = int(CH5_HQ_GRID_N_FOCUS_OPEN if n_open_hold is None else n_open_hold)
    n_hold = int(CH5_HQ_GRID_N_FOCUS_HOLD if n_focus_hold is None else n_focus_hold)
    n_fade = int(CH5_HQ_GRID_N_FOCUS_FADE if n_focus_fade is None else n_focus_fade)
    grey_w = float(CH5_HQ_GRID_FOCUS_DIM_GREY if dim_grey is None else dim_grey)
    alpha_min = float(CH5_HQ_GRID_FOCUS_DIM_ALPHA if dim_alpha is None else dim_alpha)

    frames: list = []
    with _landscape_render_context(cfg.dpi):
        mesh_prior = ch5_posterior_w12_mesh_pack(
            empty_study, empty_exam, empty_y, prior_kind=pk, config=cfg,
            z_lim=z_lim_prior,
        )
        z_ref_prior = float(np.nanmax(mesh_prior["Z"]))
        base_step = _ch5_landscape_grid_base_step(
            ws=wz, we=ez, bb=bz,
            mesh_prior=mesh_prior,
            z_lim_prior=z_lim_prior,
            z_ref_prior=z_ref_prior,
            elev=el_land,
            azim=az_land,
            pk=pk,
            z_color_lim=z_color_lim,
        )
        cells = _ch5_render_landscape_grid_cells(
            base_step, datasets, set(CH5_DATASET_KEYS), fk, per_key=per_key,
        )
        focus_kw = dict(grey_weight=grey_w, alpha_min=alpha_min)

        grid_full = ch5_composite_2x2_quadrants(cells)
        for _ in range(max(1, n_open)):
            frames.append(grid_full.copy() if hasattr(grid_full, "copy") else grid_full)
            _ch5_maybe_gc(frames)

        prev_focus: str | None = None
        for key in CH5_DATASET_KEYS:
            for tv in np.linspace(0.0, 1.0, n_fade, endpoint=True):
                frames.append(ch5_composite_2x2_focus(
                    cells, key,
                    prev_focus=prev_focus,
                    transition_u=float(tv),
                    **focus_kw,
                ))
                _ch5_maybe_gc(frames)
            lit = ch5_composite_2x2_focus(cells, key, dim_u=1.0, **focus_kw)
            for _ in range(max(1, n_hold)):
                frames.append(lit.copy() if hasattr(lit, "copy") else lit)
                _ch5_maybe_gc(frames)
            prev_focus = key

        last_key = CH5_DATASET_KEYS[-1]
        for tv in np.linspace(1.0, 0.0, n_fade, endpoint=True):
            frames.append(ch5_composite_2x2_focus(
                cells, None, dim_u=float(tv), **focus_kw,
            ))
            _ch5_maybe_gc(frames)

    return frames


def ch5_iter_uniform_landscape_grid_frames(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    n_cell_reveal_hold: int = 8,
    n_seq_hold: int = 1,
    n_annot_hold: int = 0,
    n_orbit: int | None = None,
    end_knobs_zero: bool = True,
    skip_prior_build: bool = False,
    opening_d1_zoom: bool = False,
    n_open_zoom: int | None = None,
    n_open_zoom_hold: int | None = None,
):
    """
    Yield frames for the 2×2 HQ uniform-prior grid (streaming — avoids holding all
    frames in RAM; use with ``save_mp4(generator, ...)``).
    """
    cfg = ch5_prior_landscape_config(hq=True) if config is None else config
    fk = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    pk = "uniform"
    z_lim_prior = ch5_prior_w12_z_lim(pk, scope="prior")
    z_color_lim = ch5_uniform_belief_z_lim()
    wz, ez, bz = CH5_KNOB_ZERO
    el_ct = _ch5_hq_land_elev()
    az_ct = float(_g("CH3_LIK_W12_CT_AZIM"))
    el_w1 = float(_g("CH3_LIK_W12_ELEV_W1"))
    az_w1 = float(_g("CH3_LIK_W12_AZIM_W1"))

    empty_study = np.array([], dtype=np.float64)
    empty_exam = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)

    orders = {k: [int(j) for j in datasets[k]["order"]] for k in CH5_DATASET_KEYS}
    max_pts = max(len(orders[k]) for k in CH5_DATASET_KEYS)

    n_out = 0

    def _emit(img):
        nonlocal n_out
        n_out += 1
        if n_out % 12 == 0:
            gc.collect()
        return img

    with _landscape_render_context(cfg.dpi):
        mesh_prior = ch5_posterior_w12_mesh_pack(
            empty_study, empty_exam, empty_y, prior_kind=pk, config=cfg,
            z_lim=z_lim_prior,
        )
        z_ref_prior = float(np.nanmax(mesh_prior["Z"]))

        final_mesh_by_key: dict[str, dict] = {}
        for key in CH5_DATASET_KEYS:
            ds = datasets[key]
            study = np.asarray(ds["study"], dtype=np.float64)
            exam = np.asarray(ds["exam"], dtype=np.float64)
            y_arr = np.asarray(ds["y"], dtype=np.int64)
            z_lim_final = ch5_belief_landscape_z_lim(pk, phase_u=1.0)
            final_mesh_by_key[key] = ch5_posterior_w12_mesh_pack(
                study, exam, y_arr, prior_kind=pk, config=cfg, z_lim=z_lim_final,
            )

        open_step = _LandscapeRenderStep(
            ws=wz, we=ez, bb=bz,
            elev=el_w1, azim=az_w1,
            mesh_pack=mesh_prior,
            z_lim=z_lim_prior,
            z_ref=z_ref_prior,
            study=empty_study,
            exam=empty_exam,
            y=empty_y,
            show_curves=False,
            marker=False,
            knobs_zero=True,
            prior_kind=pk,
            z_color_lim=z_color_lim,
            show_surface=False,
        )
        prior_ready_step = _LandscapeRenderStep(
            ws=wz, we=ez, bb=bz,
            elev=el_ct, azim=az_ct,
            mesh_pack=mesh_prior,
            z_lim=z_lim_prior,
            z_ref=z_ref_prior,
            study=empty_study,
            exam=empty_exam,
            y=empty_y,
            show_curves=False,
            marker=False,
            lrev=1.0,
            landscape_reveal_origin="lo_hi",
            knobs_zero=True,
            prior_kind=pk,
            z_color_lim=z_color_lim,
            show_surface=True,
            surface_grid=True,
        )
        reveal_step = prior_ready_step if skip_prior_build else open_step

        if opening_d1_zoom:
            d1_row, d1_col = _CH5_GRID_SLOTS["D1"]
            n_z_open = int(CH5_HQ_GRID_N_ZOOM if n_open_zoom is None else n_open_zoom)
            n_hold_open = int(
                CH5_HQ_GRID_N_ZOOM_HOLD if n_open_zoom_hold is None else n_open_zoom_hold
            )
            grid_d1 = _ch5_render_landscape_grid(
                prior_ready_step, datasets, {"D1"}, fk,
            )
            for _ in range(max(1, n_hold_open)):
                yield _emit(ch5_quadrant_zoom_frame(grid_d1, d1_row, d1_col, 1.0))
            for tv in np.linspace(1.0, 0.0, n_z_open, endpoint=True):
                yield _emit(ch5_quadrant_zoom_frame(grid_d1, d1_row, d1_col, float(tv)))

        visible: set[str] = set()
        if opening_d1_zoom:
            visible.add("D1")
        for key in CH5_DATASET_KEYS:
            visible.add(key)
            if opening_d1_zoom and key == "D1":
                continue
            for _ in range(max(1, int(n_cell_reveal_hold))):
                yield _emit(_ch5_render_landscape_grid(reveal_step, datasets, visible, fk))

        if not skip_prior_build:
            for step in _ch5_prior_landscape_step_stream(pk, cfg, end_knobs_zero=end_knobs_zero):
                for _ in range(max(1, int(step.hold))):
                    yield _emit(_ch5_render_landscape_grid(step, datasets, set(CH5_DATASET_KEYS), fk))

        for n in range(0, max_pts + 1):
            per_key: dict[str, dict] = {}
            for key in CH5_DATASET_KEYS:
                ds = datasets[key]
                study = np.asarray(ds["study"], dtype=np.float64)
                exam = np.asarray(ds["exam"], dtype=np.float64)
                y_arr = np.asarray(ds["y"], dtype=np.int64)
                order = orders[key]
                n_show = min(int(n), len(order))
                complete = n_show >= len(order)
                order_len = len(order)
                phase_u = 0.0 if order_len == 0 else float(n_show) / float(order_len)
                z_lim_n = ch5_belief_landscape_z_lim(pk, phase_u=phase_u)
                if complete:
                    mesh_n = final_mesh_by_key[key]
                    sn, en, yn = study, exam, y_arr
                else:
                    idxs = order[:n_show]
                    sn = study[idxs]
                    en = exam[idxs]
                    yn = y_arr[idxs]
                    mesh_n = ch5_posterior_w12_mesh_pack(
                        sn, en, yn, prior_kind=pk, config=cfg, z_lim=z_lim_n,
                    )
                extra = {
                    "mesh_pack": mesh_n,
                    "z_lim": z_lim_n,
                    "z_ref": float(np.nanmax(mesh_n["Z"])),
                    "study": sn,
                    "exam": en,
                    "y": yn,
                    "final": complete,
                }
                if complete:
                    extra.update({
                        "marker_ws": float(mesh_n["ws"]),
                        "marker_we": float(mesh_n["we"]),
                        "marker_z": float(mesh_n["marker_z"]),
                        "threshold_ws": float(mesh_n["ws"]),
                        "threshold_we": float(mesh_n["we"]),
                        "threshold_bb": float(mesh_n["bb"]),
                    })
                per_key[key] = extra

            seq_step = _LandscapeRenderStep(
                ws=wz, we=ez, bb=bz,
                elev=el_ct, azim=az_ct,
                mesh_pack=mesh_prior,
                z_lim=z_lim_prior,
                z_ref=z_ref_prior,
                study=empty_study,
                exam=empty_exam,
                y=empty_y,
                show_curves=False,
                marker=False,
                lrev=1.0,
                landscape_reveal_origin="lo_hi",
                knobs_zero=True,
                prior_kind=pk,
                z_color_lim=z_color_lim,
                show_surface=True,
                surface_grid=True,
            )
            hold_n = int(cfg.n_hold if n == 0 else n_seq_hold)
            for _ in range(max(1, hold_n)):
                yield _emit(_ch5_render_landscape_grid(
                    seq_step, datasets, set(CH5_DATASET_KEYS), fk, per_key=per_key,
                ))

        last = None
        if int(n_annot_hold) > 0:
            # Re-render one final annotated frame for hold repeats.
            per_key_final = {
                key: {
                    "mesh_pack": final_mesh_by_key[key],
                    "z_lim": ch5_belief_landscape_z_lim(pk, phase_u=1.0),
                    "z_ref": float(np.nanmax(final_mesh_by_key[key]["Z"])),
                    "study": np.asarray(datasets[key]["study"], dtype=np.float64),
                    "exam": np.asarray(datasets[key]["exam"], dtype=np.float64),
                    "y": np.asarray(datasets[key]["y"], dtype=np.int64),
                    "final": True,
                    "marker_ws": float(final_mesh_by_key[key]["ws"]),
                    "marker_we": float(final_mesh_by_key[key]["we"]),
                    "marker_z": float(final_mesh_by_key[key]["marker_z"]),
                    "threshold_ws": float(final_mesh_by_key[key]["ws"]),
                    "threshold_we": float(final_mesh_by_key[key]["we"]),
                    "threshold_bb": float(final_mesh_by_key[key]["bb"]),
                }
                for key in CH5_DATASET_KEYS
            }
            hold_step = _LandscapeRenderStep(
                ws=wz, we=ez, bb=bz,
                elev=el_ct, azim=az_ct,
                mesh_pack=mesh_prior,
                z_lim=z_lim_prior,
                z_ref=z_ref_prior,
                study=empty_study,
                exam=empty_exam,
                y=empty_y,
                show_curves=False,
                marker=False,
                lrev=1.0,
                landscape_reveal_origin="lo_hi",
                knobs_zero=True,
                prior_kind=pk,
                z_color_lim=z_color_lim,
                show_surface=True,
                surface_grid=True,
            )
            last = _ch5_render_landscape_grid(
                hold_step, datasets, set(CH5_DATASET_KEYS), fk, per_key=per_key_final,
            )
            for _ in range(int(n_annot_hold)):
                yield _emit(last)

        n_spin = int(CH5_HQ_GRID_N_ORBIT if n_orbit is None else n_orbit)
        if n_spin > 0:
            z_lim_final = ch5_belief_landscape_z_lim(pk, phase_u=1.0)
            per_key_final = {
                key: {
                    "mesh_pack": final_mesh_by_key[key],
                    "z_lim": z_lim_final,
                    "z_ref": float(np.nanmax(final_mesh_by_key[key]["Z"])),
                    "study": np.asarray(datasets[key]["study"], dtype=np.float64),
                    "exam": np.asarray(datasets[key]["exam"], dtype=np.float64),
                    "y": np.asarray(datasets[key]["y"], dtype=np.int64),
                    "final": True,
                    "marker_ws": float(final_mesh_by_key[key]["ws"]),
                    "marker_we": float(final_mesh_by_key[key]["we"]),
                    "marker_z": float(final_mesh_by_key[key]["marker_z"]),
                    "threshold_ws": float(final_mesh_by_key[key]["ws"]),
                    "threshold_we": float(final_mesh_by_key[key]["we"]),
                    "threshold_bb": float(final_mesh_by_key[key]["bb"]),
                }
                for key in CH5_DATASET_KEYS
            }
            orbit_deg = float(CH5_HQ_GRID_ORBIT_DEG)
            # endpoint=True so the last frame lands on az_ct + 360° (full turn).
            for tv in np.linspace(0.0, 1.0, n_spin, endpoint=True):
                az_spin = float(az_ct) + orbit_deg * float(tv)
                orbit_step = _LandscapeRenderStep(
                    ws=wz, we=ez, bb=bz,
                    elev=el_ct, azim=az_spin,
                    mesh_pack=mesh_prior,
                    z_lim=z_lim_prior,
                    z_ref=z_ref_prior,
                    study=empty_study,
                    exam=empty_exam,
                    y=empty_y,
                    show_curves=False,
                    marker=False,
                    lrev=1.0,
                    landscape_reveal_origin="lo_hi",
                    knobs_zero=True,
                    prior_kind=pk,
                    z_color_lim=z_color_lim,
                    show_surface=True,
                    surface_grid=True,
                )
                yield _emit(_ch5_render_landscape_grid(
                    orbit_step, datasets, set(CH5_DATASET_KEYS), fk, per_key=per_key_final,
                ))


def ch5_build_uniform_landscape_grid_frames(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    n_cell_reveal_hold: int = 8,
    n_seq_hold: int = 1,
    n_annot_hold: int = 0,
    n_orbit: int | None = None,
    end_knobs_zero: bool = True,
    skip_prior_build: bool = False,
    opening_d1_zoom: bool = False,
    n_open_zoom: int | None = None,
    n_open_zoom_hold: int | None = None,
) -> list:
    """Materialize all grid frames (may OOM on long HQ exports — prefer streaming)."""
    return list(ch5_iter_uniform_landscape_grid_frames(
        datasets,
        config=config,
        frame_kwargs=frame_kwargs,
        n_cell_reveal_hold=n_cell_reveal_hold,
        n_seq_hold=n_seq_hold,
        n_annot_hold=n_annot_hold,
        n_orbit=n_orbit,
        end_knobs_zero=end_knobs_zero,
        skip_prior_build=skip_prior_build,
        opening_d1_zoom=opening_d1_zoom,
        n_open_zoom=n_open_zoom,
        n_open_zoom_hold=n_open_zoom_hold,
    ))


def ch5_likelihood_overlay_display_z(
    log_lik,
    *,
    belief_peak: float,
    height_frac: float | None = None,
) -> np.ndarray:
    """
    Likelihood height for co-plotting under belief: exp(sum log p_i) via
    stable ``exp(ll - ll_max)``, scaled to a fraction of the belief peak.
    """
    ll = np.asarray(log_lik, dtype=float)
    frac = float(CH5_LL_OVERLAY_HEIGHT_FRAC if height_frac is None else height_frac)
    finite = ll[np.isfinite(ll)]
    if finite.size == 0:
        return np.zeros_like(ll, dtype=float)
    ll_max = float(np.max(finite))
    lik = np.exp(ll - ll_max)
    return lik * float(belief_peak) * frac


def _ch5_crop_w12_box(W1m, W2m, *arrays, w_lo: float, w_hi: float):
    """Slice mesh arrays to the axis-aligned box [w_lo, w_hi]² in (w_ST, w_EL)."""
    W1m = np.asarray(W1m, dtype=float)
    W2m = np.asarray(W2m, dtype=float)
    w1 = W1m[0, :] if W1m.ndim == 2 else np.asarray(W1m, dtype=float)
    w2 = W2m[:, 0] if W2m.ndim == 2 else np.asarray(W2m, dtype=float)
    # Detect which axis varies along rows vs cols.
    if W1m.ndim == 2 and float(np.ptp(W1m[0, :])) >= float(np.ptp(W1m[:, 0])):
        c = np.where((W1m[0, :] >= w_lo - 1e-9) & (W1m[0, :] <= w_hi + 1e-9))[0]
        r = np.where((W2m[:, 0] >= w_lo - 1e-9) & (W2m[:, 0] <= w_hi + 1e-9))[0]
    else:
        c = np.where((W1m[0, :] >= w_lo - 1e-9) & (W1m[0, :] <= w_hi + 1e-9))[0]
        r = np.where((W2m[:, 0] >= w_lo - 1e-9) & (W2m[:, 0] <= w_hi + 1e-9))[0]
        if c.size == 0 or r.size == 0:
            c = np.where((W2m[0, :] >= w_lo - 1e-9) & (W2m[0, :] <= w_hi + 1e-9))[0]
            r = np.where((W1m[:, 0] >= w_lo - 1e-9) & (W1m[:, 0] <= w_hi + 1e-9))[0]
    if c.size == 0 or r.size == 0:
        raise ValueError(f"empty crop for w in [{w_lo}, {w_hi}]")
    sl = (slice(int(r[0]), int(r[-1]) + 1), slice(int(c[0]), int(c[-1]) + 1))
    out = [W1m[sl], W2m[sl]]
    for a in arrays:
        out.append(np.asarray(a, dtype=float)[sl])
    return tuple(out)


def _ch5_diag_reveal_covers(
    ws: float,
    we: float,
    reveal_u: float,
    *,
    w1_lo: float,
    w1_hi: float,
    w2_lo: float,
    w2_hi: float,
    origin: str = "lo_hi",
) -> bool:
    """True if diagonal surface reveal ``reveal_u`` still includes (ws, we)."""
    t = float(np.clip(reveal_u, 0.0, 1.0))
    u1 = (float(ws) - float(w1_lo)) / max(float(w1_hi) - float(w1_lo), 1e-9)
    if str(origin) == "lo_hi":
        u2 = (float(w2_hi) - float(we)) / max(float(w2_hi) - float(w2_lo), 1e-9)
    else:
        u2 = (float(we) - float(w2_lo)) / max(float(w2_hi) - float(w2_lo), 1e-9)
    return (u1 + u2) <= 2.0 * t + 1e-9


def ch5_build_d1_loglik_overlay_frames(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
    n_reveal: int | None = None,
    n_hold: int | None = None,
    n_orbit: int | None = None,
) -> list:
    """
    Zoomed D1: reverse-wipe belief+mesh away → diagonal-reveal Ch4-red likelihood
    → soft belief on top → 360° orbit with vertical sprout callouts in first 90°.
    """
    cfg = ch5_prior_landscape_config(hq=True) if config is None else config
    fk = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    final = _ch5_landscape_grid_final_pack(datasets, config=cfg)
    per_key = final["per_key"]
    pk = final["prior_kind"]
    z_color_lim = final["z_color_lim"]
    extra = per_key["D1"]
    mesh = extra["mesh_pack"]
    z_lim = extra["z_lim"]
    study = extra["study"]
    exam = extra["exam"]
    y_arr = extra["y"]
    wz, ez, bz = CH5_KNOB_ZERO
    el_land = _ch5_hq_land_elev()
    az_land = float(_g("CH3_LIK_W12_CT_AZIM"))
    n_rev = int(CH5_LL_OVERLAY_N_REVEAL if n_reveal is None else n_reveal)
    n_h = int(CH5_LL_OVERLAY_N_HOLD if n_hold is None else n_hold)
    n_spin = int(CH5_LL_OVERLAY_N_ORBIT if n_orbit is None else n_orbit)
    orbit_deg = float(CH5_LL_OVERLAY_ORBIT_DEG)
    smooth = _g("ch3_knob_smoothstep")
    nll_fn = _g("_ch3_nll_sum_on_flat_grid")
    w_lo, w_hi = float(CH5_LL_OVERLAY_W_LIM[0]), float(CH5_LL_OVERLAY_W_LIM[1])

    W1m = mesh["W1m"]
    W2m = mesh["W2m"]
    B = np.full(W1m.shape, float(CH5_W12_B_FIXED), dtype=float)
    log_lik = ch5_log_likelihood_grid(study, exam, y_arr, W1m, W2m, B, nll_fn=nll_fn)
    Z_belief = np.asarray(mesh.get("Z_pdf", mesh["Z"]), dtype=float)
    belief_peak = float(np.nanmax(Z_belief))
    Z_lik_full = ch5_likelihood_overlay_display_z(log_lik, belief_peak=belief_peak)
    # Likelihood only on [-2.5, 2.5]² in (w_ST, w_EL).
    W1o, W2o, Z_lik, log_lik_box = _ch5_crop_w12_box(
        W1m, W2m, Z_lik_full, log_lik, w_lo=w_lo, w_hi=w_hi,
    )
    k = int(np.nanargmax(log_lik_box))
    mle_ws = float(W1o.ravel()[k])
    mle_we = float(W2o.ravel()[k])
    mle_z = float(Z_lik.ravel()[k])
    map_ws = float(extra["marker_ws"])
    map_we = float(extra["marker_we"])
    map_z = float(extra["marker_z"])
    # Shared floor point for best/plausible (nearly the same under uniform prior).
    peak_ws = 0.5 * (map_ws + mle_ws)
    peak_we = 0.5 * (map_we + mle_we)
    lik_rgba = (CH5_LL_OVERLAY_COLOR, float(CH5_LL_OVERLAY_ALPHA))
    reveal_origin = str(CH5_LL_OVERLAY_REVEAL_ORIGIN)
    soft_belief_alpha = float(CH5_LL_OVERLAY_BELIEF_ALPHA)
    best_label = str(CH5_LL_OVERLAY_BEST_LABEL)
    best_fig = tuple(CH5_LL_OVERLAY_BEST_LABEL_FIG)
    mesh_w1_lo = float(mesh["w1_lo"])
    mesh_w1_hi = float(mesh["w1_hi"])
    mesh_w2_lo = float(mesh["w2_lo"])
    mesh_w2_hi = float(mesh["w2_hi"])
    z_floor = float(z_lim[0])

    frames: list = []

    def _peak_covered(reveal_u: float) -> bool:
        """True while the wipe front has not yet reached the MAP peak."""
        # Strict: hide on the frame the front arrives (not after the face is gone).
        t = float(np.clip(reveal_u, 0.0, 1.0))
        u1 = (float(map_ws) - mesh_w1_lo) / max(mesh_w1_hi - mesh_w1_lo, 1e-9)
        if reveal_origin == "lo_hi":
            u2 = (mesh_w2_hi - float(map_we)) / max(mesh_w2_hi - mesh_w2_lo, 1e-9)
        else:
            u2 = (float(map_we) - mesh_w2_lo) / max(mesh_w2_hi - mesh_w2_lo, 1e-9)
        return (u1 + u2) < 2.0 * t - 1e-6

    def _emit(
        *,
        belief_lrev: float,
        lik_lrev: float = 0.0,
        belief_alpha: float | None = None,
        belief_grid: bool = False,
        belief_annot: bool = False,
        lik_annot: bool = False,
        azim: float | None = None,
        sprout_z: float | None = None,
        show_floor_dot: bool = False,
        hold: int = 1,
    ) -> None:
        b_u = float(np.clip(belief_lrev, 0.0, 1.0))
        l_u = float(np.clip(lik_lrev, 0.0, 1.0))
        az = float(az_land if azim is None else azim)
        overlay = None
        if l_u > 1e-6:
            overlay = {
                "W1m": W1o,
                "W2m": W2o,
                "Z": Z_lik,
                "reveal": l_u,
                "reveal_origin": reveal_origin,
                "rgba": lik_rgba,
                "w1_lo": w_lo,
                "w1_hi": w_hi,
                "w2_lo": w_lo,
                "w2_hi": w_hi,
                "under": True,
                "zorder": 0,
                "surface_grid": False,
                "marker": bool(lik_annot),
                "marker_ws": mle_ws,
                "marker_we": mle_we,
                "marker_z": mle_z,
                "marker_color": CH5_LL_OVERLAY_COLOR,
                "marker_edgecolors": "white",
                "here_annotation": bool(lik_annot),
                "here_label": best_label,
                "here_label_fig": best_fig,
                "here_color": CH5_LL_OVERLAY_COLOR,
                "here_text_color": "white",
            }
        sprout = None
        if show_floor_dot or (sprout_z is not None and float(sprout_z) > z_floor + 1e-6):
            sprout = {
                "ws": peak_ws,
                "we": peak_we,
                "z0": z_floor,
                "z1": float(z_floor if sprout_z is None else sprout_z),
                "show_floor_dot": bool(show_floor_dot),
                "line_color": "#111111",
                "linestyle": ":",
                "linewidth": 2.4,
            }
        img = _ch5_frame_lik_w12_belief(
            study, exam, y_arr,
            float(wz), float(ez), float(bz),
            mesh_pack=mesh,
            z_lim=z_lim,
            prior_kind=pk,
            z_color_lim=z_color_lim,
            elev=el_land,
            azim=az,
            landscape_reveal=b_u,
            landscape_reveal_origin=reveal_origin,
            show_curves=False,
            marker=bool(belief_annot),
            marker_ws=map_ws,
            marker_we=map_we,
            marker_z=map_z,
            here_annotation=bool(belief_annot),
            here_label="most plausible line",
            here_text_color="white",
            show_threshold=True,
            threshold_ws=float(extra["threshold_ws"]),
            threshold_we=float(extra["threshold_we"]),
            threshold_bb=float(extra["threshold_bb"]),
            threshold_legend_dark=False,
            z_lik_ref=float(extra["z_ref"]),
            show_surface=b_u > 1e-6,
            surface_grid=bool(belief_grid) and b_u > 1e-6,
            overlay_surface=overlay,
            belief_surface_alpha=belief_alpha,
            vertical_sprout=sprout,
            frame_kwargs=fk,
            **dict(CH5_KNOBS_UNSET_FRAME_KW),
        )
        for _ in range(max(1, int(hold))):
            frames.append(img.copy() if hasattr(img, "copy") else img)
        _ch5_maybe_gc(frames)

    with _landscape_render_context(cfg.dpi):
        # Opening: full belief (with mesh), as zoomed D1.
        _emit(
            belief_lrev=1.0, lik_lrev=0.0,
            belief_alpha=float(CH5_BELIEF_SURFACE_ALPHA),
            belief_grid=True, belief_annot=True, hold=n_h,
        )
        # 1) Clear belief + mesh; MAP marker vanishes as the wipe front reaches it.
        for tv in np.linspace(0.0, 1.0, n_rev, endpoint=True):
            u = float(smooth(float(tv)))
            b_u = 1.0 - u
            _emit(
                belief_lrev=b_u, lik_lrev=0.0,
                belief_alpha=float(CH5_BELIEF_SURFACE_ALPHA),
                belief_grid=True,
                belief_annot=_peak_covered(b_u),
            )
        _emit(
            belief_lrev=0.0, lik_lrev=0.0,
            belief_annot=False, hold=max(1, n_h // 2),
        )
        # 2) Reveal likelihood — no peak callouts yet.
        for tv in np.linspace(0.0, 1.0, n_rev, endpoint=True):
            u = float(smooth(float(tv)))
            _emit(belief_lrev=0.0, lik_lrev=u, belief_annot=False, lik_annot=False)
        _emit(
            belief_lrev=0.0, lik_lrev=1.0,
            belief_annot=False, lik_annot=False, hold=n_h,
        )
        # 3) Reveal soft belief on top — still no peak callouts.
        for tv in np.linspace(0.0, 1.0, n_rev, endpoint=True):
            u = float(smooth(float(tv)))
            _emit(
                belief_lrev=u, lik_lrev=1.0,
                belief_alpha=soft_belief_alpha,
                belief_grid=False,
                belief_annot=False, lik_annot=False,
            )
        _emit(
            belief_lrev=1.0, lik_lrev=1.0,
            belief_alpha=soft_belief_alpha,
            belief_grid=False,
            belief_annot=False, lik_annot=False, hold=n_h,
        )
        # 4) 360° orbit: first 90° floor-dot + vertical sprout to plausible height;
        #    reveal best-line then most-plausible callouts as the line passes each z.
        for tv in np.linspace(0.0, 1.0, n_spin, endpoint=True):
            t = float(tv)
            az = az_land + orbit_deg * t
            if t <= 0.25 + 1e-12:
                # Linear rise so mle_z / map_z thresholds land mid-quarter, not delayed.
                sprout_u = float(np.clip(t / 0.25, 0.0, 1.0))
            else:
                sprout_u = 1.0
            z_line = z_floor + sprout_u * (map_z - z_floor)
            _emit(
                belief_lrev=1.0, lik_lrev=1.0,
                belief_alpha=soft_belief_alpha,
                belief_grid=False,
                belief_annot=z_line >= map_z - 1e-6,
                lik_annot=z_line >= mle_z - 1e-6,
                azim=az,
                sprout_z=z_line,
                show_floor_dot=True,
            )
        _emit(
            belief_lrev=1.0, lik_lrev=1.0,
            belief_alpha=soft_belief_alpha,
            belief_grid=False,
            belief_annot=True, lik_annot=True,
            azim=az_land + orbit_deg,
            sprout_z=map_z,
            show_floor_dot=True,
            hold=n_h,
        )

    return frames


# --- HITA parallel FrameSpec APIs (preserved on sync from FULL) ---

def _landscape_step_spec(
    *,
    ws: float,
    we: float,
    bb: float,
    elev: float,
    azim: float,
    mesh_key: str,
    z_lim: tuple[float, float],
    z_ref: float | None = None,
    prior_kind: str = "uniform",
    z_color_lim: tuple[float, float] | None = None,
    knobs_zero: bool = False,
    show_surface: bool = True,
    surface_grid: bool = False,
    lrev: float = 0.0,
    landscape_reveal_origin: str = "lo_lo",
    squish_u: float = 0.0,
) -> dict:
    return {
        "ws": float(ws),
        "we": float(we),
        "bb": float(bb),
        "elev": float(elev),
        "azim": float(azim),
        "mesh_key": str(mesh_key),
        "z_lim": (float(z_lim[0]), float(z_lim[1])),
        "z_ref": z_ref,
        "prior_kind": prior_kind,
        "z_color_lim": z_color_lim,
        "knobs_zero": bool(knobs_zero),
        "show_surface": bool(show_surface),
        "surface_grid": bool(surface_grid),
        "lrev": float(lrev),
        "landscape_reveal_origin": landscape_reveal_origin,
        "squish_u": float(squish_u),
    }


def _landscape_step_from_spec(step_spec: dict, pack: dict) -> _LandscapeRenderStep:
    meshes = pack["meshes"]
    mesh_key = step_spec["mesh_key"]
    mesh_pack = meshes[mesh_key]
    z_ref = step_spec.get("z_ref")
    if z_ref is None:
        z_ref = float(np.nanmax(mesh_pack["Z"]))
    return _LandscapeRenderStep(
        ws=step_spec["ws"],
        we=step_spec["we"],
        bb=step_spec["bb"],
        elev=step_spec["elev"],
        azim=step_spec["azim"],
        mesh_pack=mesh_pack,
        z_lim=step_spec["z_lim"],
        z_ref=float(z_ref),
        knobs_zero=step_spec.get("knobs_zero", False),
        prior_kind=step_spec.get("prior_kind", "uniform"),
        z_color_lim=step_spec.get("z_color_lim"),
        show_surface=step_spec.get("show_surface", True),
        surface_grid=step_spec.get("surface_grid", False),
        lrev=step_spec.get("lrev", 0.0),
        landscape_reveal_origin=step_spec.get("landscape_reveal_origin", "lo_lo"),
        squish_u=step_spec.get("squish_u", 0.0),
    )


def _per_key_from_spec(per_key_spec: dict, pack: dict) -> dict[str, dict]:
    meshes = pack["meshes"]
    out: dict[str, dict] = {}
    passthrough = (
        "data_xlim",
        "data_ylim",
        "zero_axis_guides_u",
        "show_shadow_marker",
        "shadow_marker_ws",
        "shadow_marker_we",
        "shadow_marker_z",
        "show_shadow_threshold",
        "shadow_threshold_ws",
        "shadow_threshold_we",
        "shadow_threshold_bb",
        "here_annotation",
        "origin_guides",
    )
    for key, extra in per_key_spec.items():
        mesh_key = extra.get("mesh_key")
        if mesh_key is None:
            continue
        mesh_n = meshes[mesh_key]
        ds = pack["datasets"][key]
        study_full = np.asarray(ds["study"], dtype=np.float64)
        exam_full = np.asarray(ds["exam"], dtype=np.float64)
        y_full = np.asarray(ds["y"], dtype=np.int64)
        # Sequential reveal: prefer explicit n_show (2D icons match posterior mesh).
        if "n_show" in extra:
            order = [int(j) for j in ds["order"]]
            n_show = int(min(max(0, int(extra["n_show"])), len(order)))
            idxs = order[:n_show]
            study_vis = study_full[idxs]
            exam_vis = exam_full[idxs]
            y_vis = y_full[idxs]
        else:
            study_vis = np.asarray(extra.get("study", study_full), dtype=np.float64)
            exam_vis = np.asarray(extra.get("exam", exam_full), dtype=np.float64)
            y_vis = np.asarray(extra.get("y", y_full), dtype=np.int64)
        entry = {
            "mesh_pack": mesh_n,
            "z_lim": extra.get("z_lim", pack.get("z_lim_final")),
            "z_ref": float(extra.get("z_ref", np.nanmax(mesh_n["Z"]))),
            "study": study_vis,
            "exam": exam_vis,
            "y": y_vis,
            "final": bool(extra.get("final", False)),
        }
        if extra.get("marker") or extra.get("final"):
            entry.update({
                "marker_ws": float(mesh_n["ws"]),
                "marker_we": float(mesh_n["we"]),
                "marker_z": float(mesh_n["marker_z"]),
                "threshold_ws": float(mesh_n["ws"]),
                "threshold_we": float(mesh_n["we"]),
                "threshold_bb": float(mesh_n["bb"]),
            })
        for field in passthrough:
            if field in extra:
                entry[field] = extra[field]
        out[key] = entry
    return out


def ch5_render_landscape_grid_from_spec(pack: dict, spec: dict):
    """Top-level picklable render: one 2×2 landscape grid frame."""
    step = _landscape_step_from_spec(spec["step"], pack)
    visible = set(spec.get("visible_keys", CH5_DATASET_KEYS))
    per_key = _per_key_from_spec(spec.get("per_key", {}), pack)
    fk = pack.get("frame_kwargs")
    img = _ch5_render_landscape_grid(step, pack["datasets"], visible, fk, per_key=per_key or None)
    zoom = spec.get("zoom_cell")
    if zoom is not None:
        row, col, zu = zoom
        img = ch5_quadrant_zoom_frame(img, int(row), int(col), float(zu))
    return img


def ch5_build_uniform_landscape_grid_pack(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
) -> dict:
    """Precompute meshes for parallel ch5_47 export."""
    cfg = ch5_prior_landscape_config(hq=True) if config is None else config
    fk = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    pk = "uniform"
    z_lim_prior = ch5_prior_w12_z_lim(pk, scope="prior")
    z_color_lim = ch5_uniform_belief_z_lim()
    empty_study = np.array([], dtype=np.float64)
    empty_exam = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)

    meshes: dict[str, dict] = {}
    mesh_prior = ch5_posterior_w12_mesh_pack(
        empty_study, empty_exam, empty_y, prior_kind=pk, config=cfg, z_lim=z_lim_prior,
    )
    meshes["__mesh_prior__"] = mesh_prior

    orders = {k: [int(j) for j in datasets[k]["order"]] for k in CH5_DATASET_KEYS}
    max_pts = max(len(orders[k]) for k in CH5_DATASET_KEYS)
    z_lim_final = ch5_belief_landscape_z_lim(pk, phase_u=1.0)

    for key in CH5_DATASET_KEYS:
        ds = datasets[key]
        study = np.asarray(ds["study"], dtype=np.float64)
        exam = np.asarray(ds["exam"], dtype=np.float64)
        y_arr = np.asarray(ds["y"], dtype=np.int64)
        meshes[f"__final_mesh_{key}__"] = ch5_posterior_w12_mesh_pack(
            study, exam, y_arr, prior_kind=pk, config=cfg, z_lim=z_lim_final,
        )

    for n in range(0, max_pts + 1):
        for key in CH5_DATASET_KEYS:
            ds = datasets[key]
            study = np.asarray(ds["study"], dtype=np.float64)
            exam = np.asarray(ds["exam"], dtype=np.float64)
            y_arr = np.asarray(ds["y"], dtype=np.int64)
            order = orders[key]
            n_show = min(int(n), len(order))
            complete = n_show >= len(order)
            if complete:
                continue
            phase_u = 0.0 if len(order) == 0 else float(n_show) / float(len(order))
            z_lim_n = ch5_belief_landscape_z_lim(pk, phase_u=phase_u)
            idxs = order[:n_show]
            meshes[f"__seq_mesh_{key}_{n}__"] = ch5_posterior_w12_mesh_pack(
                study[idxs], exam[idxs], y_arr[idxs],
                prior_kind=pk, config=cfg, z_lim=z_lim_n,
            )

    return {
        "config": cfg,
        "frame_kwargs": fk,
        "datasets": datasets,
        "meshes": meshes,
        "z_lim_prior": z_lim_prior,
        "z_lim_final": z_lim_final,
        "z_color_lim": z_color_lim,
        "prior_kind": pk,
    }


def ch5_iter_uniform_landscape_grid_specs(
    pack: dict,
    *,
    n_cell_reveal_hold: int = 8,
    n_seq_hold: int = 1,
    n_annot_hold: int = 0,
    n_orbit: int | None = None,
    opening_d1_zoom: bool = True,
    n_open_zoom: int | None = None,
    n_open_zoom_hold: int | None = None,
    hold_tail: int = 0,
    clip_id: str = "ch5_47",
) -> Iterator[dict]:
    """Yield picklable spec dicts for parallel render (no images)."""
    cfg = pack["config"]
    datasets = pack["datasets"]
    pk = pack["prior_kind"]
    z_lim_prior = pack["z_lim_prior"]
    z_color_lim = pack["z_color_lim"]
    wz, ez, bz = CH5_KNOB_ZERO
    el_ct = _ch5_hq_land_elev()
    az_ct = float(_g("CH3_LIK_W12_CT_AZIM"))
    el_w1 = float(_g("CH3_LIK_W12_ELEV_W1"))
    az_w1 = float(_g("CH3_LIK_W12_AZIM_W1"))
    orders = {k: [int(j) for j in datasets[k]["order"]] for k in CH5_DATASET_KEYS}
    max_pts = max(len(orders[k]) for k in CH5_DATASET_KEYS)
    mesh_prior_key = "__mesh_prior__"
    z_ref_prior = float(np.nanmax(pack["meshes"][mesh_prior_key]["Z"]))

    prior_ready = _landscape_step_spec(
        ws=wz, we=ez, bb=bz, elev=el_ct, azim=az_ct,
        mesh_key=mesh_prior_key, z_lim=z_lim_prior, z_ref=z_ref_prior,
        knobs_zero=True, prior_kind=pk, z_color_lim=z_color_lim,
        show_surface=True, surface_grid=True, lrev=1.0,
        landscape_reveal_origin="lo_hi",
    )
    reveal_step = prior_ready

    n_z_open = int(CH5_HQ_GRID_N_ZOOM if n_open_zoom is None else n_open_zoom)
    n_hold_open = int(CH5_HQ_GRID_N_ZOOM_HOLD if n_open_zoom_hold is None else n_open_zoom_hold)
    d1_row, d1_col = _CH5_GRID_SLOTS["D1"]

    if opening_d1_zoom:
        for _ in range(max(1, n_hold_open)):
            yield {
                "clip_id": clip_id, "kind": "landscape_grid",
                "step": prior_ready, "visible_keys": ("D1",), "per_key": {},
                "zoom_cell": (d1_row, d1_col, 1.0),
            }
        for tv in np.linspace(1.0, 0.0, n_z_open, endpoint=True):
            yield {
                "clip_id": clip_id, "kind": "landscape_grid",
                "step": prior_ready, "visible_keys": ("D1",), "per_key": {},
                "zoom_cell": (d1_row, d1_col, float(tv)),
            }

    visible: list[str] = []
    if opening_d1_zoom:
        visible.append("D1")
    for key in CH5_DATASET_KEYS:
        visible.append(key)
        if opening_d1_zoom and key == "D1":
            continue
        for _ in range(max(1, int(n_cell_reveal_hold))):
            yield {
                "clip_id": clip_id, "kind": "landscape_grid",
                "step": reveal_step, "visible_keys": tuple(visible), "per_key": {},
            }

    for n in range(0, max_pts + 1):
        per_key: dict[str, dict] = {}
        for key in CH5_DATASET_KEYS:
            order = orders[key]
            n_show = min(int(n), len(order))
            complete = n_show >= len(order)
            mesh_key = f"__final_mesh_{key}__" if complete else f"__seq_mesh_{key}_{n}__"
            phase_u = 0.0 if len(order) == 0 else float(n_show) / float(len(order))
            z_lim_n = ch5_belief_landscape_z_lim(pk, phase_u=phase_u)
            extra: dict = {
                "mesh_key": mesh_key,
                "final": complete,
                # Keep 2D icons in lockstep with the sequential posterior mesh.
                "n_show": int(n_show),
                "z_lim": z_lim_n if not complete else pack["z_lim_final"],
            }
            if complete:
                extra["marker"] = True
                extra["threshold"] = True
            per_key[key] = extra
        hold_n = int(cfg.n_hold if n == 0 else n_seq_hold)
        for _ in range(max(1, hold_n)):
            yield {
                "clip_id": clip_id, "kind": "landscape_grid",
                "step": prior_ready, "visible_keys": tuple(CH5_DATASET_KEYS),
                "per_key": per_key,
            }

    if int(n_annot_hold) > 0:
        per_key_final = {
            k: {"mesh_key": f"__final_mesh_{k}__", "final": True, "marker": True, "threshold": True}
            for k in CH5_DATASET_KEYS
        }
        for _ in range(int(n_annot_hold)):
            yield {
                "clip_id": clip_id, "kind": "landscape_grid",
                "step": prior_ready, "visible_keys": tuple(CH5_DATASET_KEYS),
                "per_key": per_key_final,
            }

    n_spin = int(CH5_HQ_GRID_N_ORBIT if n_orbit is None else n_orbit)
    last_spec: dict | None = None
    if n_spin > 0:
        orbit_deg = float(CH5_HQ_GRID_ORBIT_DEG)
        per_key_final = {
            k: {"mesh_key": f"__final_mesh_{k}__", "final": True, "marker": True, "threshold": True}
            for k in CH5_DATASET_KEYS
        }
        for tv in np.linspace(0.0, 1.0, n_spin, endpoint=False):
            az_spin = float(az_ct) + orbit_deg * float(tv)
            orbit_step = _landscape_step_spec(
                ws=wz, we=ez, bb=bz, elev=el_ct, azim=az_spin,
                mesh_key=mesh_prior_key, z_lim=z_lim_prior, z_ref=z_ref_prior,
                knobs_zero=True, prior_kind=pk, z_color_lim=z_color_lim,
                show_surface=True, surface_grid=True, lrev=1.0,
                landscape_reveal_origin="lo_hi",
            )
            last_spec = {
                "clip_id": clip_id, "kind": "landscape_grid",
                "step": orbit_step, "visible_keys": tuple(CH5_DATASET_KEYS),
                "per_key": per_key_final,
            }
            yield last_spec

    if last_spec is None:
        # fallback: hold on final annotated grid if orbit disabled
        per_key_final = {
            k: {"mesh_key": f"__final_mesh_{k}__", "final": True, "marker": True, "threshold": True}
            for k in CH5_DATASET_KEYS
        }
        last_spec = {
            "clip_id": clip_id, "kind": "landscape_grid",
            "step": prior_ready, "visible_keys": tuple(CH5_DATASET_KEYS),
            "per_key": per_key_final,
        }

    for _ in range(int(hold_tail)):
        yield dict(last_spec)


def ch5_build_grid_2d_zoom_shadow_orbit_pack(
    datasets: dict[str, dict],
    *,
    config: Ch5LandscapeConfig | None = None,
    frame_kwargs: dict | None = None,
) -> dict:
    """Prior + final meshes only (no sequential partials) for clips 54/56."""
    cfg = ch5_grid_landscape_config() if config is None else config
    fk = _ch5_lik_w12_frame_kwargs(frame_kwargs)
    pk = "uniform"
    z_lim_prior = ch5_prior_w12_z_lim(pk, scope="prior")
    z_lim_final = ch5_belief_landscape_z_lim(pk, phase_u=1.0)
    z_color_lim = ch5_uniform_belief_z_lim()
    empty_study = np.array([], dtype=np.float64)
    empty_exam = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)

    meshes: dict[str, dict] = {}
    mesh_prior = ch5_posterior_w12_mesh_pack(
        empty_study, empty_exam, empty_y, prior_kind=pk, config=cfg, z_lim=z_lim_prior,
    )
    meshes["__mesh_prior__"] = mesh_prior
    for key in CH5_DATASET_KEYS:
        ds = datasets[key]
        study = np.asarray(ds["study"], dtype=np.float64)
        exam = np.asarray(ds["exam"], dtype=np.float64)
        y_arr = np.asarray(ds["y"], dtype=np.int64)
        meshes[f"__final_mesh_{key}__"] = ch5_posterior_w12_mesh_pack(
            study, exam, y_arr, prior_kind=pk, config=cfg, z_lim=z_lim_final,
        )

    return {
        "config": cfg,
        "frame_kwargs": fk,
        "datasets": datasets,
        "meshes": meshes,
        "z_lim_prior": z_lim_prior,
        "z_lim_final": z_lim_final,
        "z_color_lim": z_color_lim,
        "prior_kind": pk,
    }


def ch5_iter_grid_2d_zoom_shadow_orbit_specs(
    pack: dict,
    *,
    n_zoom: int | None = None,
    n_hold: int | None = None,
    n_orbit: int | None = None,
    n_guide: int | None = None,
    n_cam: int | None = None,
    n_d4_swing: int | None = None,
    target_lim: tuple[float, float] | None = None,
    camera_pan: bool = False,
    hold_tail: int = 0,
    clip_id: str = "ch5_54",
) -> Iterator[dict]:
    """Yield picklable landscape_grid specs for ch5_54 / ch5_56 (no images)."""
    pk = pack["prior_kind"]
    z_lim_prior = pack["z_lim_prior"]
    z_color_lim = pack["z_color_lim"]
    meshes = pack["meshes"]
    wz, ez, bz = CH5_KNOB_ZERO
    el_land = _ch5_hq_land_elev()
    el_top = float(CH5_GRID_2D_CAM_TOP_ELEV)
    do_cam = bool(camera_pan)
    el_shadow = el_top if do_cam else el_land
    az_base = float(_g("CH3_LIK_W12_CT_AZIM"))
    mesh_prior_key = "__mesh_prior__"
    z_ref_prior = float(np.nanmax(meshes[mesh_prior_key]["Z"]))

    n_z = int(CH5_GRID_2D_ZOOM_N if n_zoom is None else n_zoom)
    n_h = int(CH5_GRID_2D_ZOOM_N_HOLD if n_hold is None else n_hold)
    n_o = int(CH5_GRID_2D_ORBIT_N if n_orbit is None else n_orbit)
    n_g = int(CH5_GRID_2D_GUIDE_N if n_guide is None else n_guide)
    n_c = int(CH5_GRID_2D_CAM_N if n_cam is None else n_cam)
    n_sw = int(CH5_GRID_2D_D4_N_SWING if n_d4_swing is None else n_d4_swing)
    tlo, thi = (
        float(CH5_GRID_2D_ZOOM_TARGET[0]),
        float(CH5_GRID_2D_ZOOM_TARGET[1]),
    ) if target_lim is None else (float(target_lim[0]), float(target_lim[1]))
    r_max = float(CH5_GRID_2D_ORBIT_R_MAX)
    r_min = float(CH5_GRID_2D_ORBIT_R_MIN)
    d4_dw_st = float(CH5_GRID_2D_D4_DW_ST)
    d4_dw_el = float(CH5_GRID_2D_D4_DW_EL)

    start_lims = {
        # Shared (0, 7) window for all four cells — D3's custom meta lims would
        # desync the zoom choreography from D1/D2/D4.
        key: (tuple(CH5_STANDARD_XLIM), tuple(CH5_STANDARD_YLIM))
        for key in CH5_DATASET_KEYS
    }
    map_xy: dict[str, tuple[float, float, float]] = {}
    orbit_geom: dict[str, tuple[float, float, float]] = {}
    for key in CH5_DATASET_KEYS:
        mesh_n = meshes[f"__final_mesh_{key}__"]
        mws = float(mesh_n["ws"])
        mwe = float(mesh_n["we"])
        mbb = float(mesh_n["bb"])
        map_xy[key] = (mws, mwe, mbb)
        r_nat = float(np.hypot(mws, mwe))
        th0 = float(np.arctan2(mwe, mws)) if r_nat > 1e-6 else 0.25 * np.pi
        if r_nat < 0.15:
            r_use = r_min
        else:
            r_use = min(r_nat, r_max)
        orbit_geom[key] = (r_use, th0, r_nat)

    d4_mws, d4_mwe, _d4_bb = map_xy["D4"]
    d4_out_ws = float(np.clip(d4_mws + d4_dw_st, -r_max, r_max))
    d4_out_we = float(np.clip(d4_mwe + d4_dw_el, -r_max, r_max))
    d4_orbit_r = float(np.hypot(d4_out_ws, d4_out_we))
    d4_orbit_th0 = float(np.arctan2(d4_out_we, d4_out_ws))

    smooth = _g("ch3_knob_smoothstep")
    lerp = _g("ch3_lerp")
    last_spec: dict | None = None

    def _shadow_state(key: str, ws_s: float, we_s: float) -> dict:
        mws, mwe, mbb = map_xy[key]
        moved = abs(ws_s - mws) > 1e-5 or abs(we_s - mwe) > 1e-5
        return {
            "show_shadow_marker": moved,
            "show_shadow_threshold": moved,
            "shadow_marker_ws": float(ws_s),
            "shadow_marker_we": float(we_s),
            "shadow_threshold_ws": float(ws_s),
            "shadow_threshold_we": float(we_s),
            "shadow_threshold_bb": float(mbb),
        }

    def _d4_swing_shadows(u: float) -> dict[str, dict]:
        ws_s = float(lerp(d4_mws, d4_out_ws, u))
        we_s = float(lerp(d4_mwe, d4_out_we, u))
        return {"D4": _shadow_state("D4", ws_s, we_s)}

    def _orbit_shadows(theta: float) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for key in CH5_DATASET_KEYS:
            mws, mwe, _ = map_xy[key]
            wrap = abs(theta) < 1e-8 or abs(theta - 2.0 * np.pi) < 1e-8
            if key == "D4":
                ws_s = d4_orbit_r * float(np.cos(d4_orbit_th0 + theta))
                we_s = d4_orbit_r * float(np.sin(d4_orbit_th0 + theta))
                if wrap:
                    ws_s, we_s = d4_out_ws, d4_out_we
            else:
                r_use, th0, r_nat = orbit_geom[key]
                ws_s = r_use * float(np.cos(th0 + theta))
                we_s = r_use * float(np.sin(th0 + theta))
                if wrap and r_nat >= 1e-6 and abs(r_use - r_nat) < 1e-6:
                    ws_s, we_s = mws, mwe
            out[key] = _shadow_state(key, ws_s, we_s)
        return out

    def _make_spec(
        *,
        elev: float,
        panel_u: float,
        guides_u: float,
        shadows: dict[str, dict] | None = None,
    ) -> dict:
        per_key: dict[str, dict] = {}
        for key in CH5_DATASET_KEYS:
            (x0, x1), (y0, y1) = start_lims[key]
            entry: dict[str, Any] = {
                "mesh_key": f"__final_mesh_{key}__",
                "final": True,
                "marker": True,
                "threshold": True,
                "here_annotation": True,
                "data_xlim": _ch5_lerp_limits(x0, x1, tlo, thi, panel_u),
                "data_ylim": _ch5_lerp_limits(y0, y1, tlo, thi, panel_u),
                "zero_axis_guides_u": float(guides_u),
                "show_shadow_marker": False,
                "show_shadow_threshold": False,
            }
            if shadows and key in shadows:
                entry.update(shadows[key])
            per_key[key] = entry
        step = _landscape_step_spec(
            ws=wz, we=ez, bb=bz,
            elev=float(elev), azim=az_base,
            mesh_key=mesh_prior_key, z_lim=z_lim_prior, z_ref=z_ref_prior,
            knobs_zero=True, prior_kind=pk, z_color_lim=z_color_lim,
            show_surface=True, surface_grid=True, lrev=1.0,
            landscape_reveal_origin="lo_hi",
        )
        return {
            "clip_id": clip_id,
            "kind": "landscape_grid",
            "step": step,
            "visible_keys": tuple(CH5_DATASET_KEYS),
            "per_key": per_key,
        }

    def _emit(
        *,
        elev: float,
        panel_u: float,
        guides_u: float,
        shadows: dict[str, dict] | None = None,
        hold: int = 1,
    ):
        nonlocal last_spec
        spec = _make_spec(
            elev=elev, panel_u=panel_u, guides_u=guides_u, shadows=shadows,
        )
        last_spec = spec
        for _ in range(max(1, int(hold))):
            yield dict(spec)

    yield from _emit(elev=el_land, panel_u=0.0, guides_u=0.0, hold=max(1, n_h // 2))

    for tv in np.linspace(0.0, 1.0, n_g, endpoint=True):
        yield from _emit(
            elev=el_land, panel_u=0.0, guides_u=float(smooth(float(tv))),
        )

    yield from _emit(elev=el_land, panel_u=0.0, guides_u=1.0, hold=n_h)

    for tv in np.linspace(0.0, 1.0, n_z, endpoint=True):
        u = float(smooth(float(tv)))
        yield from _emit(elev=el_land, panel_u=u, guides_u=1.0)

    yield from _emit(elev=el_land, panel_u=1.0, guides_u=1.0, hold=n_h)

    if do_cam:
        for tv in np.linspace(0.0, 1.0, n_c, endpoint=True):
            u = float(smooth(float(tv)))
            yield from _emit(
                elev=float(lerp(el_land, el_top, u)),
                panel_u=1.0, guides_u=1.0,
            )
        yield from _emit(
            elev=el_top, panel_u=1.0, guides_u=1.0, hold=max(1, n_h // 2),
        )

    for tv in np.linspace(0.0, 1.0, n_sw, endpoint=True):
        u = float(smooth(float(tv)))
        yield from _emit(
            elev=el_shadow, panel_u=1.0, guides_u=1.0,
            shadows=_d4_swing_shadows(u),
        )

    yield from _emit(
        elev=el_shadow, panel_u=1.0, guides_u=1.0,
        shadows=_d4_swing_shadows(1.0), hold=max(1, n_h // 2),
    )

    for tv in np.linspace(0.0, 1.0, n_o, endpoint=True):
        theta = 2.0 * np.pi * float(tv)
        yield from _emit(
            elev=el_shadow, panel_u=1.0, guides_u=1.0,
            shadows=_orbit_shadows(theta),
        )

    yield from _emit(
        elev=el_shadow, panel_u=1.0, guides_u=1.0,
        shadows=_orbit_shadows(2.0 * np.pi), hold=max(1, n_h // 2),
    )

    for tv in np.linspace(0.0, 1.0, n_sw, endpoint=True):
        u = float(smooth(float(tv)))
        yield from _emit(
            elev=el_shadow, panel_u=1.0, guides_u=1.0,
            shadows=_d4_swing_shadows(1.0 - u),
        )

    yield from _emit(
        elev=el_shadow, panel_u=1.0, guides_u=1.0, hold=max(1, n_h // 2),
    )

    if do_cam:
        for tv in np.linspace(0.0, 1.0, n_c, endpoint=True):
            u = float(smooth(float(tv)))
            yield from _emit(
                elev=float(lerp(el_top, el_land, u)),
                panel_u=1.0 - u, guides_u=1.0,
            )

    yield from _emit(
        elev=el_land,
        panel_u=0.0 if do_cam else 1.0,
        guides_u=1.0,
        hold=n_h,
    )

    if last_spec is not None:
        for _ in range(int(hold_tail)):
            yield dict(last_spec)
