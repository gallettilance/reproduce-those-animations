"""Chapter 6 — full story builders and ordered export registry."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image

from ch5_core import (
    CH5_BELIEF_SURFACE_ALPHA,
    CH5_CREDIBLE_MASS,
    CH5_CT_GRID,
    CH5_CT_N_PIVOT,
    CH5_CT_N_SWEEP,
    CH5_DATASET_KEYS,
    CH5_GRID,
    CH5_MS,
    CH5_N_DECOMP,
    CH5_N_HOLD,
    CH5_N_SEQ_HOLD,
    CH5_PLANE_ALPHA,
    CH5_PRIOR_LAND_MS,
    CH5_HQ_LAND_MS,
    CH5_STEM_SURF_MS,
    CH5_HQ_N_ANNOT,
    CH5_HQ_N_FADE,
    CH5_HQ_N_TEXT_FADE,
    CH5_HQ_N_HIST3D,
    CH5_HQ_N_LAYOUT,
    CH5_HQ_N_SEQ_HOLD,
    CH5_HQ_CT_GRID,
    CH5_HQ_CT_N_HOLD,
    CH5_HQ_CT_N_PIVOT,
    CH5_HQ_CT_N_SWEEP,
    CH5_HQ_GRID_FOCUS_DIM_ALPHA,
    CH5_HQ_GRID_FOCUS_DIM_GREY,
    CH5_HQ_GRID_N_CAM_MORPH,
    CH5_HQ_GRID_N_FOCUS_FADE,
    CH5_HQ_GRID_N_ORBIT,
    CH5_HQ_GRID_N_SQUISH,
    CH5_HQ_GRID_N_ZOOM,
    CH5_HQ_GRID_N_ZOOM_HOLD,
    CH5_HQ_GRID_N_ZOOM_ORBIT,
    CH5_HQ_GRID_ORBIT_DEG,
    CH5_CT_VOXEL_TOUR_N_ERASE,
    CH5_CT_VOXEL_TOUR_N_HOLD,
    CH5_CT_VOXEL_TOUR_N_MAP,
    CH5_CT_VOXEL_TOUR_N_BOX,
    CH5_CT_VOXEL_TOUR_N_VOXEL_SWAP,
    CH5_CRED_WANDER_N_HOLD,
    CH5_CRED_WANDER_N_MORPH,
    CH5_CRED_WANDER_N_SEG,
    CH5_CRED_WANDER_PROBE_COLOR,
    CH5_CRED_WANDER_THRESHOLD_COLOR,
    CH5_CRED_WANDER_THRESHOLD_LW,
    CH5_CRED_WANDER_INTERVAL_MARK_SIZE,
    CH5_CRED_WANDER_INTERVAL_MARK_ALPHA,
    CH5_CRED_THICK_N_ROT,
    CH5_CRED_THICK_ROT_DEG,
    CH5_CRED_THICK_N_SEG,
    CH5_CRED_THICK_SCALE_STEP,
    CH5_CRED_AXIS_N_SEG,
    CH5_CRED_AXIS_N_HOLD,
    CH5_MAP_QUIVER_LEN,
    CH5_MAP_QUIVER_LENGTHS,
    CH5_MAP_QUIVER_LW,
    CH5_MAP_QUIVER_N_ORBIT,
    CH5_MAP_QUIVER_N_HOLD,
    CH5_MAP_QUIVER_N_SLIDE,
    CH5_MAP_QUIVER_COLORS,
    CH5_MAP_QUIVER_DOMINANCE,
    CH5_MAP_QUIVER_BLACK,
    CH5_ELLIPSOID_LAYER_SCALES,
    CH5_ELLIPSOID_FACE_ALPHAS,
    CH5_ELLIPSOID_EDGE_ALPHAS,
    CH5_ELLIPSOID_N_LAYER,
    CH5_ELLIPSOID_N_ORBIT,
    CH5_ELLIPSOID_N_HOLD,
    CH5_ELLIPSOID_MESH_U,
    CH5_ELLIPSOID_MESH_V,
    CH5_ELLIP_FROM64_N_FADE,
    CH5_ELLIP_FROM64_N_QUIVER,
    CH5_D3_SIG_N_HOLD,
    CH5_D3_SIG_N_CROSS,
    CH5_D3_SIG_N_TILT,
    CH5_D3_SIG_N_RAISE,
    CH5_D3_SIG_N_SURF,
    CH5_D3_SIG_N_ORBIT,
    CH5_D3_SIG_ELEV_FLAT,
    CH5_D3_SIG_ELEV_3D,
    CH5_D3_SIG_AZ_TOP,
    CH5_D3_SIG_AZ0,
    CH5_D3_SIG_MESH_N,
    CH5_D3_SIG_TOP_SCALE,
    CH5_D3_SIG_TOP_DY,
    CH5_D3_SIG_LEFT_ZOOM,
    CH5_D3_SIG_LEFT_ZOOM_ORBIT,
    CH5_D3_SIG_MESH_PAD,
    CH5_D3_SIG_ZLIM,
    CH5_D3_SIG_ORBIT_ELEV_DELTA,
    CH5_D3_SIG_OVERLAP_WIDTH,
    CH5_D3_SIG_OVERLAP_HEIGHT,
    CH5_D3_SIG_RIGHT_SHRINK,
    CH5_D3_SIG_TILT_DX,
    CH5_D3_SIG_LEFT_GROW,
    CH5_D3_SIG_GROW_START,
    CH5_D3_SIG_BAND_STEP,
    CH5_D3_SIG_BAND_OVERLAP,
    CH5_D3_SIG_ICON_SPAN_FRAC,
    CH5_D3_SIG_ICON_SPAN_FRAC_POST,
    CH5_D1_FAN_N_HOLD,
    CH5_D1_FAN_N_MOVE,
    CH5_D1_FAN_N_ADD_MORPH,
    CH5_D1_FAN_N_ADD_HOLD,
    CH5_D1_FAN_ADD_BATCH,
    CH5_D1_FAN_N_SETUP_ROT,
    CH5_D1_FAN_SETUP_SPIN,
    CH5_D1_FAN_N_ORBIT,
    CH5_D1_FAN_ORBIT_DEG,
    CH5_D1_FAN_MOVE_TO,
    CH5_D1_FAN_ADDS,
    CH5_D2_NOISE_EMPH_N_HOLD,
    CH5_D2_NOISE_EMPH_N_GROW,
    CH5_D2_NOISE_EMPH_SCALE,
    CH5_D2_NOISE_PASS,
    CH5_D2_NOISE_FAIL,
    CH5_BEST_LINE_N_HOLD,
    CH5_BEST_LINE_N_THRESH,
    CH5_BEST_LINE_N_CELEBRATE,
    CH5_BEST_LINE_N_GREY_LOGO,
    CH5_BEST_LINE_N_FADE,
    CH5_BEST_LINE_N_D4_CUT,
    CH5_BEST_LINE_N_D4_CUTS,
    CH5_BEST_LINE_N_PARALLEL,
    CH5_BEST_LINE_PARALLEL_BIAS,
    CH5_BEST_LINE_LOGO_FRAC,
    CH5_BEST_LINE_LOGO_CX,
    CH5_BEST_LINE_LW_THICK,
    CH5_BEST_LINE_LW_NORMAL,
    CH5_BEST_LINE_LW_THIN,
    CH5_BEST_LINE_N_LAYOUT,
    CH5_BEST_LINE_N_LAND_REVEAL,
    CH5_KNOBS_UNSET_FRAME_KW,
    CH5_LL_OVERLAY_REVEAL_ORIGIN,
    CH5_KNOB_ZERO,
    CH5_PRIOR_TEX,
    CH5_LIK_TEX,
    CH5_BAYES_TEX,
    CH5_TEXT_LAST_ID,
    CH5_VIEW_BOUNDS,
    CH5_VOXEL_GRID,
    CH5_VOXEL_N_FILL,
    CH5_VOXEL_N_ROT,
    CH5_VOXEL_N_PROJ_SHADOW,
    CH5_VOXEL_N_PROJ_COLLAPSE,
    CH5_VOXEL_ORBIT_DEG,
    CH5_VOXEL_N_ORBIT,
    CH5_W12_B_FIXED,
    _CH3_DRAFT,
    ch5_all_dataset_packs,
    ch5_calibrate_global_density_scales,
    ch5_dataset_pack,
    ch5_density_from_log,
    ch5_global_density_limits,
    ch5_hq_land_elev,
    ch5_log_likelihood_grid,
    ch5_log_posterior_grid,
    ch5_log_prior,
    ch5_log_prior_grid,
    ch5_posterior_pdf_from_log,
    ch5_posterior_w12_pdf,
    ch5_prior_3d_pdf,
    ch5_belief_w12_pdf,
    ch5_belief_landscape_z_lim,
    ch5_clip_belief_height,
    ch5_prior_w12_pdf_peak,
    ch5_prior_w12_log_flat,
    ch5_prior_w12_belief_peak,
    ch5_prior_w12_z_lim,
    ch5_posterior_display_density,
    ch5_plot_belief_surface_with_grid,
    ch5_surface_grid_plot_kw,
    ch5_posterior_grid_weights,
    ch5_posterior_3d_pack,
    ch5_posterior_map_continuous,
    ch5_posterior_w12_surface,
    ch5_prior_w12_z_lim,
    ch5_reveal_order,
    ch5_w12_mesh,
    ch5_hessian_eigen_frame,
)
from ch5_datasets import CH5_D4_CENTER, CH5_DATASET_META, ch5_plot_limits, ch5_unpack_dataset
from ch5_layout import (
    ch5_composite_2x2_focus,
    ch5_composite_2x2_quadrants,
    ch5_crossfade_images,
    ch5_dim_cell_image,
    ch5_overlay_howithink_center_right,
    ch5_confetti_best_line_overlay,
    ch5_draw_hpd_orthogonal_projection,
    ch5_draw_hpd_point_cloud,
    ch5_draw_interval_parallelepiped,
    ch5_draw_map_parameter_annotation,
    ch5_draw_map_parameter_marker,
    ch5_draw_hpd_voxels_fill,
    ch5_draw_landscape_origin_guides,
    ch5_blend_param_colors,
    ch5_eigen_quiver_color_plan,
    ch5_eigen_quiver_newton_colors,
    ch5_hessian_cell_colors_for_axes,
    ch5_draw_map_basis_quivers,
    ch5_draw_laplace_ellipsoid,
    ch5_laplace_ellipsoid_radii,
    ch5_draw_zero_axis_cross_2d,
    ch5_draw_zero_axis_cross_3d,
    ch5_fig_to_image,
    ch5_figure_grid,
    ch5_hpd_region_on_floor,
    ch5_hpd_voxel_diag_projection,
    ch5_place_howithink_badge,
    ch5_quadrant_zoom_frame,
    ch5_uniform_belief_facecolors,
    ch5_uniform_belief_z_lim,
    _ch5_hpd_cell_centers,
)
from ch4_layout import (
    CH4_DUO_FIGSIZE,
    CH4_COMPOSER,
    ch4_blocks_write_from_slot,
    ch4_bottom_per_block_progress,
    ch4_formula_hessian_matrix_block,
)
import ch5_prior_landscape
from ch6_layout import CH6_FIGSIZE
from latex_to_handwrite import draw_handwrite_matrix_in_cell

_G: dict[str, Any] = {}
_PACKS: dict[str, dict] = {}

# Three competing hypotheses — order: purple (left), blue (middle), yellow (right) on 2D plot.
CH5_THREE_LINES = (
    (1.0, -1.0, -4.0),
    (1.0, -1.0, 0.0),
    (1.0, -1.0, 4.0),
)
CH5_THREE_LINE_COLORS = ("#9966cc", "#5577bb", "#dd8833")
CH5_THREE_LINE_LABELS = (r"$L_1$", r"$L_2$", r"$L_3$")
CH5_THREE_LINE_PRIOR = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
CH5_SEQ_BAR_YMAX = 1.05
CH5_SEQ_BAR_YMAX_POST = 1.05
CH5_SEQ_BAR_LABEL_FS = 16.0
CH5_SEQ_BAR_WIDTH = 0.52
CH5_SEQ_BAR_SLOTS = (0, 1, 2)
# Histogram slots left→right: yellow, blue, purple (2D line indices unchanged).
CH5_SEQ_BAR_LINE_IDX = (2, 1, 0)
# Plot + right-rail reveal order matches histogram left→right.
CH5_SEQ_LINE_REVEAL_IDX = (2, 1, 0)
CH5_D1_SEQ_START_IDX = 9  # (3, 2) pass — first point in sequential-bars story
# D4: four iso-posterior hypotheses (similar log-posterior, far from MAP at origin).
CH5_D4_ALT_LINES = (
    (0.15, 0.15, -0.975),
    (-0.75, 0.75, -0.075),
    (0.675, -0.675, -0.225),
    (-0.15, -0.15, 0.975),
)
CH5_ALT_LINE_COLORS = ("#c45c3a", "#3a9e6a", "#3a5fc4", "#9e3a9e")
CH5_BIAS_SWEEP_DELTA = 1.0
CH5_BIAS_SWEEP_N = 20
# Intro → ch5_01: two extra competing hypotheses (visible on the D1 panel).
CH5_INTRO_EXTRA_LINES = (
    (0.90, -1.10, 2.5),
    (1.10, -0.90, -3.5),
)
CH5_FIVE_LINES = CH5_THREE_LINES + CH5_INTRO_EXTRA_LINES
CH5_FIVE_LINE_COLORS = CH5_THREE_LINE_COLORS + ("#2a8a4a", "#8a3a5a")
CH5_FIVE_LINE_PRIOR = np.full(5, 1.0 / 5.0)
CH5_D1_VARIANT_SHIFTS = (
    (0.35, -0.25),
    (-0.30, 0.35),
    (0.20, 0.20),
    (-0.25, -0.20),
)
CH5_INTRO_N_VARIANTS = 4
CH5_INTRO_N_DUO_MORPH = 18
CH5_INTRO_N_LAYOUT_MORPH = 22
CH5_INTRO_N_PANEL_MORPH = 14
# ch4_02a/02b wide 2D panel — same axes slot as chapter 4 transitions.
CH5_CH4_WIDE_DATA = (0.06, 0.14, 0.88, 0.72)
_CH5_INTRO_DUO_RECTS: tuple | None = None
_CH5_SEQ_BARS_DUO_RECTS: tuple | None = None
_CH5_WEIGHT3D_DUO_RECTS: tuple | None = None
CH5_HQ_LAND_FRAME_KW = {"show_colormap": False, "show_legend": False}
CH5_SEQ_RIGHT_GAP_SCALE = 0.20
CH5_SEQ_CONTENT_LIFT_IN = 1.0
CH5_SEQ_RIGHT_TEXT_X_FRAC = 0.015
CH5_SEQ_LINE_TITLE_FS = 28.0
CH5_SEQ_LINE_VAL_FS = 20.0
CH5_SEQ_LINE_VAL_LINE_DY_PT = 4.2
CH5_SEQ_LINE_VAL_EXTRA_DY_PT = 2.8
CH5_SEQ_LINE_TITLE_VAL_GAP_PT = -5.0
CH5_SEQ_LINE_GROUP_GAP_PT = 1.2
CH5_SEQ_BOTTOM_BAYES_EXTRA_DROP_PT = 12.0
# Fraction form is taller (num + bar + denom); lift so denom clears the frame edge.
CH5_SEQ_BAYES_FRAC_LIFT_PT = 16.0
CH5_SEQ_BAYES_FS = 23.0
CH5_SEQ_LIK_Y_INSET_PT = -3.0
CH5_SEQ_LIK_X_SHIFT_PT = -10.0
CH5_SEQ_BAYES_TEXT_X_FRAC = 0.02
CH5_SEQ_PROB_X_SHIFT_EXTRA_PT = -20.0
CH5_SEQ_CORNER_TITLE_LIFT_IN = 3.0 / 25.4 + 0.18
# Display order for exact-values rail: Line 1 yellow, Line 2 blue, Line 3 purple.
CH5_SEQ_LINE_DISPLAY = (
    {"slot": 2, "num": 1, "color": "#dd8833", "name": "Line 1"},
    {"slot": 1, "num": 2, "color": "#5577bb", "name": "Line 2"},
    {"slot": 0, "num": 3, "color": "#9966cc", "name": "Line 3"},
)
CH5_SEQ_DATA_LIK_TEX = (
    r"$P(\mathrm{Data} \mid \mathrm{line}) = "
    r"\prod_{i=0}^{N} p(y_i \mid x_i)$"
)
CH5_SEQ_BAYES_FRAC_TEX = (
    r"$P(\mathrm{line} \mid \mathrm{Data}) = "
    r"\frac{P(\mathrm{line})\, P(\mathrm{Data} \mid \mathrm{line})}{P(\mathrm{Data})}$"
)
CH5_SEQ_BAYES_NEW_FRAC_TEX = (
    r"$New Belief = "
    r"\frac{P(\mathrm{line})\, P(\mathrm{Data} \mid \mathrm{line})}{P(\mathrm{Data})}$"
)
_CH5_SEQ_COMPOSERS: dict[str, Any] = {}
_CH5_SEQ_PLOT_CACHE: dict[tuple, Image.Image] = {}
_CH5_SEQ_COMPOSE_CACHE: dict[tuple, Image.Image] = {}
_CH5_SEQ_SHELL_KEY = "ch5_01_seq"
_CH5_SEQ_RAILS_KEY = "ch5_01_seq_r"


def _ch5_seq_layout(*, lift_in: float | None = None, right_gap_scale: float | None = None):
    from ch4_layout import CH4_CANVAS_H_IN, CH4_LAYOUT, CH4_RIGHT_TEXT_SHIFT_IN

    lift = float(CH5_SEQ_CONTENT_LIFT_IN if lift_in is None else lift_in)
    lift_frac = lift / float(CH4_CANVAS_H_IN)
    gap_scale = float(CH5_SEQ_RIGHT_GAP_SCALE if right_gap_scale is None else right_gap_scale)
    return replace(
        CH4_LAYOUT,
        right_section_drop_frac=CH4_LAYOUT.right_section_drop_frac - lift_frac * 0.20 + 0.010,
        right_rail_lower_frac=max(0.0, CH4_LAYOUT.right_rail_lower_frac - lift_frac * 0.50),
        bottom_section_drop_frac=CH4_LAYOUT.bottom_section_drop_frac,
        bottom_text_drop_frac=CH4_LAYOUT.bottom_text_drop_frac,
        bottom_content_lift_frac=0.0,
        corner_title_lift_in=CH5_SEQ_CORNER_TITLE_LIFT_IN,
        right_row_gap_frac=CH4_LAYOUT.right_row_gap_frac * gap_scale,
        right_text_shift_in=max(0.0, CH4_RIGHT_TEXT_SHIFT_IN - 0.22),
    )


def _ch5_seq_composer(variant: str = "balanced"):
    from ch4_layout import CH4_EXPORT, make_composer

    if variant not in _CH5_SEQ_COMPOSERS:
        presets = {
            "compact": dict(lift_in=0.85, right_gap_scale=0.18),
            "balanced": dict(lift_in=1.0, right_gap_scale=0.20),
            "airy": dict(lift_in=1.15, right_gap_scale=0.24),
        }
        kw = presets.get(variant, presets["balanced"])
        _CH5_SEQ_COMPOSERS[variant] = make_composer(
            "classic_light", export=CH4_EXPORT, layout=_ch5_seq_layout(**kw),
        )
    return _CH5_SEQ_COMPOSERS[variant]


def _ch5_reload_lik_w12_landscape_inc() -> None:
    """Ensure ``ch3_frame_lik_w12_3d`` matches the blender .inc source (always reload)."""
    from pathlib import Path

    inc_path = Path(__file__).resolve().parent / "blender" / "ch4_02_likelihood_w12_landscape.inc"
    if inc_path.is_file():
        exec(compile(inc_path.read_text(), str(inc_path), "exec"), _G)
    _G["_ch5_lik_inc_reloaded"] = True


def install(globals_dict: dict[str, Any]) -> None:
    global _G, _PACKS
    _G = globals_dict
    _G["ch5_plot_belief_surface_with_grid"] = ch5_plot_belief_surface_with_grid
    _G["ch5_draw_landscape_origin_guides"] = ch5_draw_landscape_origin_guides
    _G["ch5_draw_zero_axis_cross_2d"] = ch5_draw_zero_axis_cross_2d
    _G["ch5_draw_zero_axis_cross_3d"] = ch5_draw_zero_axis_cross_3d
    _ch5_reload_lik_w12_landscape_inc()
    nll_fn = _G["_ch3_nll_sum_on_flat_grid"]

    # Chapter artifact cache: density scales + dataset packs
    art = None
    try:
        from hita.config.profile import active_profile
        from hita.config.series import SeriesConfig
        from hita.export.cache import code_hash, get_chapter_artifacts, put_chapter_artifacts
        from hita.export.context import project_root

        art_key = code_hash(("ch5_core", "ch5_datasets", "ch5_story"))
        art_key = f"{SeriesConfig.load().version}_{active_profile().name}_{art_key}"
        art = get_chapter_artifacts(5, art_key, project_root())
    except Exception:
        art = None

    packs_ok = (
        isinstance(art, dict)
        and isinstance(art.get("packs"), dict)
        and all(k in art["packs"] for k in CH5_DATASET_KEYS)
        and isinstance(art.get("density"), dict)
    )
    if packs_ok:
        import ch5_core as c5c

        dens = art["density"]
        c5c.CH5_GLOBAL_LOG_POST_REF = dens.get("CH5_GLOBAL_LOG_POST_REF", c5c.CH5_GLOBAL_LOG_POST_REF)
        if "CH5_BELIEF_W12_Z_HI" in dens:
            c5c.CH5_BELIEF_W12_Z_HI = dens["CH5_BELIEF_W12_Z_HI"]
        if "CH5_BELIEF_W12_Z_HI_D1" in dens:
            c5c.CH5_BELIEF_W12_Z_HI_D1 = dens["CH5_BELIEF_W12_Z_HI_D1"]
        _PACKS = art["packs"]
        print("  ch5 context cache: hit (packs + density scales)", flush=True)
    else:
        ch5_calibrate_global_density_scales(nll_fn=nll_fn)
        _PACKS = ch5_all_dataset_packs(
            nll_fn=nll_fn,
            nll_grad_fn=_G.get("_ch3_nll_sum_grad_at_point"),
        )
        try:
            import ch5_core as c5c
            from hita.config.profile import active_profile
            from hita.config.series import SeriesConfig
            from hita.export.cache import code_hash, put_chapter_artifacts
            from hita.export.context import project_root

            art_key = code_hash(("ch5_core", "ch5_datasets", "ch5_story"))
            art_key = f"{SeriesConfig.load().version}_{active_profile().name}_{art_key}"
            put_chapter_artifacts(
                5,
                art_key,
                {
                    "packs": _PACKS,
                    "density": {
                        "CH5_GLOBAL_LOG_POST_REF": dict(c5c.CH5_GLOBAL_LOG_POST_REF),
                        "CH5_BELIEF_W12_Z_HI": getattr(c5c, "CH5_BELIEF_W12_Z_HI", None),
                        "CH5_BELIEF_W12_Z_HI_D1": getattr(c5c, "CH5_BELIEF_W12_Z_HI_D1", None),
                    },
                },
                project_root(),
            )
        except Exception:
            pass

    _ch5_install_empty_panel_patch()
    ch5_prior_landscape.install(_G)
    _CH5_SEQ_COMPOSERS.clear()


def _ensure_packs() -> None:
    """Rebuild dataset packs if a module reload left ``_PACKS`` empty."""
    global _PACKS
    if isinstance(_PACKS, dict) and all(k in _PACKS for k in CH5_DATASET_KEYS):
        return
    if not _G or "_ch3_nll_sum_on_flat_grid" not in _G:
        raise RuntimeError(
            "ch5_story packs missing and chapter globals unset — "
            "re-run the Chapter 5 setup cell"
        )
    nll_fn = _G["_ch3_nll_sum_on_flat_grid"]
    print("  ch5_story: rebuilding dataset packs (_PACKS was empty)", flush=True)
    ch5_calibrate_global_density_scales(nll_fn=nll_fn)
    _PACKS = ch5_all_dataset_packs(
        nll_fn=nll_fn,
        nll_grad_fn=_G.get("_ch3_nll_sum_grad_at_point"),
    )


def _g(name: str):
    return _G[name]


def _ch5_style_empty_data_panel(ax, *, xlim, ylim):
    """Axis labels/ticks matching ``draw_dataset`` when no points are shown yet."""
    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    fs = float(_g("AXIS_LABEL_SIZE"))
    ax.set_xlabel("Study time (hours)", fontsize=fs, labelpad=10)
    ax.set_ylabel("Exam length (hours)", fontsize=fs, labelpad=10)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", which="major", labelsize=float(_g("FONT_SIZE")))
    _g("finalize_style_legend_tex")(ax)


def _ch5_install_empty_panel_patch():
    """Skip σ colormap + decision line when ch3 left panel has no data points."""
    if _G.get("_ch5_patched_draw_left_panel"):
        return
    _orig = _G["ch3_draw_left_panel"]

    def _patched(ax, w_st, w_el, b, study, exam, y, legend_label, **kw):
        if np.asarray(study).size == 0:
            bxlim = kw.get("boundary_xlim", _G["xlim"])
            bylim = kw.get("boundary_ylim", _G["ylim"])
            _ch5_style_empty_data_panel(ax, xlim=bxlim, ylim=bylim)
            return
        return _orig(ax, w_st, w_el, b, study, exam, y, legend_label, **kw)

    _G["ch3_draw_left_panel"] = _patched
    _G["_ch5_patched_draw_left_panel"] = True


def _ch5_visible_count(study, exam, y, *, mask=None) -> int:
    if mask is not None:
        return int(np.sum(np.asarray(mask, dtype=bool)))
    return int(len(np.asarray(study)))


def _ch5_plot_threshold_line(
    ax,
    ws,
    we,
    bb,
    xlim,
    ylim,
    *,
    color="grey",
    linestyle="--",
    linewidth=1.8,
    alpha=0.85,
    label=None,
    legend_only_if_missing=False,
):
    """Decision boundary segment clipped to the 2D panel."""
    ws, we, bb = float(ws), float(we), float(bb)
    if abs(ws) < 1e-6 and abs(we) < 1e-6:
        if label is not None:
            ax.plot(
                [], [], c=color, linestyle=linestyle, linewidth=linewidth,
                alpha=alpha, label=label,
            )
        return False
    bxy = _g("boundary_line_xy")(ws, we, bb, float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1]))
    if bxy is not None:
        bx, by = bxy
        ax.plot(
            bx, by, c=color, linestyle=linestyle, linewidth=linewidth,
            alpha=alpha, label=label, zorder=3,
        )
        return True
    if legend_only_if_missing and label is not None:
        ax.plot([], [], c=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)
    return False


def _ch5_draw_data_panel(
    ax,
    study,
    exam,
    y,
    *,
    xl,
    yl,
    mask=None,
    ws=None,
    we=None,
    bb=None,
    show_colormap=False,
    show_threshold=False,
    extra_thresholds=None,
    legend_emphasis="st",
    threshold_legend=True,
    threshold_linewidth=None,
    threshold_linestyle="--",
    threshold_color="grey",
    after_draw=None,
):
    """2D roster: σ colormap and single threshold only when points are visible."""
    study, exam, y = np.asarray(study), np.asarray(exam), np.asarray(y)
    n_vis = _ch5_visible_count(study, exam, y, mask=mask)
    if mask is not None:
        draw_mask = np.asarray(mask, dtype=bool)
    else:
        draw_mask = None

    if n_vis == 0:
        if len(y) > 0:
            _g("draw_dataset")(ax, study, exam, y, mask=np.zeros(len(y), dtype=bool))
        if after_draw is not None:
            after_draw(ax, xl, yl)
        _ch5_style_empty_data_panel(ax, xlim=xl, ylim=yl)
        return

    if show_colormap and ws is not None:
        leg = _g("legend_linear_equation_values_bold_param")(ws, we, bb, legend_emphasis)
        _g("ch3_draw_left_panel")(
            ax, ws, we, bb, study, exam, y, leg,
            show_colormap=True, highlight_mistakes_flag=False,
            **_ch5_colormap_panel_kw(xl, yl),
        )
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
    else:
        _g("draw_dataset")(ax, study, exam, y, mask=draw_mask)
        drew_legend = False
        want_leg = bool(threshold_legend)
        if show_threshold and ws is not None:
            leg = (
                _g("legend_linear_equation_values_bold_param")(ws, we, bb, legend_emphasis)
                if want_leg else None
            )
            lw = float(CH5_BEST_LINE_LW_NORMAL if threshold_linewidth is None else threshold_linewidth)
            _ch5_plot_threshold_line(
                ax, ws, we, bb, xl, yl,
                color=str(threshold_color), linestyle=str(threshold_linestyle),
                linewidth=lw, label=leg, legend_only_if_missing=True,
            )
            drew_legend = bool(leg)
        if extra_thresholds:
            for item in extra_thresholds:
                ews, ewe, ebb = float(item[0]), float(item[1]), float(item[2])
                color = item[3] if len(item) > 3 else "#888888"
                linestyle = item[4] if len(item) > 4 else "-"
                linewidth = float(item[5]) if len(item) > 5 else 1.6
                alpha = float(item[6]) if len(item) > 6 else 0.80
                leg = (
                    _g("legend_linear_equation_values_bold_param")(ews, ewe, ebb, legend_emphasis)
                    if want_leg else None
                )
                _ch5_plot_threshold_line(
                    ax, ews, ewe, ebb, xl, yl,
                    color=color, linestyle=linestyle, linewidth=linewidth,
                    alpha=alpha, label=leg,
                )
                if leg:
                    drew_legend = True
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        if drew_legend and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", prop={"size": _G["LEGEND_SIZE"]})
        _g("finalize_style_legend_tex")(ax)
    if after_draw is not None:
        after_draw(ax, xl, yl)


def _ch5_sigma_mesh(xlim, ylim):
    """σ colormap mesh over the visible 2D panel (same resolution as ST_KNOB / EL_KNOB)."""
    ref_st, ref_el = _g("ST_KNOB"), _g("EL_KNOB")
    st = np.linspace(float(xlim[0]), float(xlim[1]), int(ref_st.shape[1]))
    el = np.linspace(float(ylim[0]), float(ylim[1]), int(ref_el.shape[0]))
    return np.meshgrid(st, el)


def _ch5_colormap_panel_kw(xlim, ylim):
    sigma_stg, sigma_elg = _ch5_sigma_mesh(xlim, ylim)
    return dict(
        boundary_xlim=xlim,
        boundary_ylim=ylim,
        sigma_stg=sigma_stg,
        sigma_elg=sigma_elg,
    )


def _hold(img, n=None):
    n = CH5_N_HOLD if n is None else int(n)
    return [img] * max(1, n)


def _draft_range(n_full, n_draft):
    return int(n_draft if _CH3_DRAFT else n_full)


def _use_text(clip_id: str) -> bool:
    return clip_id <= CH5_TEXT_LAST_ID


def _formula_blocks():
    return [
        _g("_ch4_formula_hand_block")(CH5_PRIOR_TEX, formula_slot="prior", weight=0.28, bold_lhs=True),
        _g("_ch4_formula_hand_block")(CH5_LIK_TEX, formula_slot="lik", weight=0.36, bold_lhs=True, text_x_frac=0.04),
        _g("_ch4_formula_hand_block")(CH5_BAYES_TEX, formula_slot="bayes", weight=0.36, bold_lhs=False, text_x_frac=0.04),
    ]


def _density_cmap_limits(*, prior_kind="gaussian"):
    return ch5_global_density_limits(prior_kind)


def _density_legend_blocks(*, prior_kind="gaussian"):
    lo, hi = _density_cmap_limits(prior_kind=prior_kind)
    blocks = _g("ch4_nll_heatmap_legend_blocks")(lo, hi)
    # Bar runs dark (bottom) → pink (top); labels match inverted density map.
    blocks[0]["label_lo"] = f"{hi:.1f}  (high)"
    blocks[0]["label_hi"] = f"{lo:.1f}  (low)"
    blocks[0]["nll_lo"] = lo
    blocks[0]["nll_hi"] = hi
    return blocks


def _density_facecolors(density, *, invert=True, prior_kind="gaussian"):
    """Belief cmap on the shared Ch4 heatmap scale (global pdf peak across datasets)."""
    pk = str(prior_kind).lower()
    if pk == "uniform":
        return ch5_uniform_belief_facecolors(
            density, z_lim=ch5_uniform_belief_z_lim(), prior_kind=pk,
            surface_alpha=CH5_BELIEF_SURFACE_ALPHA,
        )
    lo, hi = _density_cmap_limits(prior_kind=pk)
    d = ch5_clip_belief_height(density, prior_kind=pk, z_lim=(lo, hi))
    span = max(hi - lo, 1e-12)
    normed = np.clip((d - lo) / span, 0.0, 1.0)
    t = 1.0 - normed if invert else normed
    return _g("ch4_nll_heatmap_cmap")()(t)


def _draw_density_surface(
    ax3d, W1, W2, density, *, colored=True, mono=False, squish_u=0.0, alpha=None, invert_cmap=True,
    prior_red=False, prior_kind="gaussian", z_lim=None,
):
    """Posterior / density surface — Ch4 heatmap, solid Ch4 red (mono / prior), or colored."""
    pk = str(prior_kind).lower()
    if z_lim is None:
        z_lim = ch5_prior_w12_z_lim(pk)
    al = float(CH5_PLANE_ALPHA if alpha is None else alpha)
    z = ch5_clip_belief_height(density, prior_kind=pk, z_lim=z_lim)
    u = float(np.clip(squish_u, 0.0, 1.0))
    z_show = z * (1.0 - u)
    if pk == "uniform" and (prior_red or colored):
        face = ch5_uniform_belief_facecolors(
            z, z_lim=ch5_uniform_belief_z_lim(), prior_kind=pk,
            surface_alpha=CH5_BELIEF_SURFACE_ALPHA,
        )
        ch5_plot_belief_surface_with_grid(
            ax3d, W1, W2, z_show, facecolors=face, alpha=al, zorder=5,
        )
    elif prior_red or mono or not colored:
        lik_a = float(_g("CH3_LIK_W12_SURFACE_ALPHA"))
        ch5_plot_belief_surface_with_grid(
            ax3d, W1, W2, z_show, color=_g("CH4_LIK_SURFACE_COLOR"),
            alpha=lik_a, zorder=5,
        )
    else:
        face = _density_facecolors(z, invert=invert_cmap, prior_kind=pk)
        ch5_plot_belief_surface_with_grid(
            ax3d, W1, W2, z_show, facecolors=face, alpha=al, zorder=5,
        )


def _ch5_ct_grid_n():
    """Full-size CT slice resolution — match Ch4 when globals are installed."""
    try:
        return int(_g("CH3_LIK_CT_GRID"))
    except KeyError:
        return int(CH5_CT_GRID)


def _ch5_mesh_pdf(log_post, w1, w2):
    """Pdf normalized on a 2-D parameter mesh (slice or w_ST–w_EL grid)."""
    w1a = np.asarray(w1, dtype=np.float64)
    w2a = np.asarray(w2, dtype=np.float64)
    return ch5_belief_w12_pdf(
        log_post,
        w1_lo=float(np.nanmin(w1a)),
        w1_hi=float(np.nanmax(w1a)),
        w2_lo=float(np.nanmin(w2a)),
        w2_hi=float(np.nanmax(w2a)),
    )


def _ch5_ct_slice_pdf(log_post, w1, w2, b):
    """
    Posterior pdf on a CT slice mesh, normalized over the plane's two free axes.

    Works for axis-aligned and pivoted planes (where one of w_ST/w_EL/b may be
    constant, so a naive w_ST–w_EL cell area would be zero).
    """
    from ch5_core import ch5_posterior_pdf_from_log

    w1a = np.asarray(w1, dtype=np.float64)
    w2a = np.asarray(w2, dtype=np.float64)
    ba = np.asarray(b, dtype=np.float64)
    lp = np.asarray(log_post, dtype=np.float64)
    if lp.ndim != 2 or w1a.shape != lp.shape:
        # Fall back to peak-normalized shape if mesh is unexpected.
        return ch5_density_from_log(lp)
    # Parallelogram area of one grid cell in (w_ST, w_EL, b) space.
    e_i = np.array([
        w1a[1, 0] - w1a[0, 0],
        w2a[1, 0] - w2a[0, 0],
        ba[1, 0] - ba[0, 0],
    ], dtype=np.float64)
    e_j = np.array([
        w1a[0, 1] - w1a[0, 0],
        w2a[0, 1] - w2a[0, 0],
        ba[0, 1] - ba[0, 0],
    ], dtype=np.float64)
    cell = float(np.linalg.norm(np.cross(e_i, e_j)))
    if not np.isfinite(cell) or cell <= 1e-18:
        return ch5_density_from_log(lp)
    return ch5_posterior_pdf_from_log(lp, cell)


def _ch5_ct_view_init(ax3d, *, cam_azim_u=0.0, cam_spin_deg=0.0, hq_elev=False):
    """CT camera — optional HQ elev offset (canonical CT elev − 10°)."""
    _g("ch4_lik_ct_view_init")(ax3d, cam_azim_u=float(cam_azim_u), cam_spin_deg=float(cam_spin_deg))
    if hq_elev:
        base = float(_g("CH3_LIK_CH4_CT_ELEV"))
        ax3d.view_init(elev=ch5_hq_land_elev(base), azim=float(ax3d.azim))


def _ch5_ct_slice_map_point(ws, we, bb, sweep_axis, plane_val):
    """Global MAP (w_ST, w_EL, b) projected onto an axis-aligned CT slice plane."""
    axis = str(sweep_axis).lower()
    v = float(plane_val)
    ws, we, bb = float(ws), float(we), float(bb)
    if axis == "b":
        return ws, we, v
    if axis == "st":
        return v, we, bb
    return ws, v, bb


def _ch5_ct_slice_map_from_mesh(w1, w2, b, log_post):
    """Most-plausible point on the current CT mesh (always lies on the plane)."""
    lp = np.asarray(log_post, dtype=float).ravel()
    if lp.size == 0 or not np.any(np.isfinite(lp)):
        return 0.0, 0.0, 0.0
    k = int(np.nanargmax(lp))
    return (
        float(np.asarray(w1, dtype=float).ravel()[k]),
        float(np.asarray(w2, dtype=float).ravel()[k]),
        float(np.asarray(b, dtype=float).ravel()[k]),
    )


def _ch5_ct_slice_has_plausible_mass(log_post) -> bool:
    """True iff the slice has any finite posterior (inside the plausible region)."""
    lp = np.asarray(log_post, dtype=float)
    return bool(np.any(np.isfinite(lp)))


def _ch5_ct_draw_map_on_slice(fig, ax3d, px, py, pz, *, label="most plausible line"):
    """Mark the on-plane MAP and label it like the 47+ landscapes."""
    z_lo, z_hi = ax3d.get_zlim()
    z_span = max(float(z_hi) - float(z_lo), 1e-12)
    z_mark = float(pz) + float(_g("CH3_LIK_MARKER_Z_BUMP_FRAC")) * z_span
    mcol = _g("FAIL_COLOR")
    ax3d.scatter(
        [float(px)], [float(py)], [z_mark],
        color=mcol, edgecolors="white", linewidths=1.8,
        s=220.0, depthshade=False, zorder=30,
    )
    _g("_ch3_lik_w12_here_annotation")(
        fig, ax3d, float(px), float(py), z_mark,
        label=str(label),
        color=mcol, text_color="white", edgecolor="white",
    )


def _ch5_ct_draw_slice(ax3d, w1, w2, b, log_post, *, log_prior=None, prior_red=False, prior_kind="gaussian"):
    """CT slice in (w_ST, w_EL, b) space — density as face color, not as z-height."""
    from matplotlib.colors import to_rgba

    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    b = np.asarray(b, dtype=float)
    pk = str(prior_kind).lower()
    # Match 47+ landscape transparency for uniform; keep Ch4 plane alpha otherwise.
    al = float(CH5_BELIEF_SURFACE_ALPHA if pk == "uniform" else _g("CH3_LIK_CT_PLANE_ALPHA"))
    if prior_red:
        rgba = to_rgba(_g("CH4_LIK_SURFACE_COLOR"), alpha=al)
        face = np.empty(w1.shape + (4,), dtype=float)
        face[..., :] = rgba
    else:
        density = _ch5_ct_slice_pdf(log_post, w1, w2, b)
        if pk == "uniform":
            face = ch5_uniform_belief_facecolors(
                density,
                z_lim=ch5_uniform_belief_z_lim(),
                surface_alpha=al,
            )
        else:
            density = ch5_clip_belief_height(density, prior_kind=pk)
            face = _density_facecolors(density, prior_kind=pk)
            face[..., 3] = al
    ax3d.plot_surface(
        w1, w2, b,
        facecolors=face,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        zorder=6,
    )


def _ch5_ct_pivot_mesh_b_st(bounds, *, theta_u, gn):
    """
    Rotate the horizontal b=lo plane into the vertical st=lo plane.
    Shared edge: (w_ST=lo, w_EL, b=lo) — continuous handoff after scanning down b.
    """
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = bounds
    u = float(np.clip(float(theta_u), 0.0, 1.0))
    u = float(_g("ch3_knob_smoothstep")(u))
    ang = u * (np.pi / 2.0)
    lob = float(dlob)
    lo_st = float(dlo1)
    w1s, w2, _bs = _g("_ch4_ct_sweep_mesh")("b", lob, bounds, gn=int(gn))
    dw = np.asarray(w1s, dtype=float) - lo_st
    w1 = lo_st + np.cos(ang) * dw
    b = lob + np.sin(ang) * dw
    return w1, w2, b


def _ch5_ct_mesh_and_log_post(
    study,
    exam,
    y,
    bounds,
    *,
    sweep_axis=None,
    plane_val=0.0,
    pivot_from=None,
    pivot_to=None,
    pivot_u=0.0,
    prior_kind="gaussian",
    gn=None,
):
    gn = int(_ch5_ct_grid_n() if gn is None else gn)
    pf = None if pivot_from is None else str(pivot_from).lower()
    pt = None if pivot_to is None else str(pivot_to).lower()
    if pf is not None and pt is not None:
        if pf == "b" and pt == "st":
            w1, w2, b = _ch5_ct_pivot_mesh_b_st(bounds, theta_u=float(pivot_u), gn=gn)
        else:
            w1, w2, b, _ = _g("_ch4_ct_pivot_mesh")(
                study, exam, y, pf, pt, bounds,
                theta_u=float(pivot_u), gn=gn,
            )
    else:
        w1, w2, b = _g("_ch4_ct_sweep_mesh")(str(sweep_axis), float(plane_val), bounds, gn=gn)
    log_post = ch5_log_posterior_grid(
        study, exam, y, w1, w2, b,
        nll_fn=_g("_ch3_nll_sum_on_flat_grid"), prior_kind=str(prior_kind),
    )
    log_prior = ch5_log_prior_grid(w1, w2, b, kind=str(prior_kind))
    return w1, w2, b, log_post, log_prior


def _ch5_style_belief_w12_ax3d(ax3d, *, prior_kind="gaussian", z_lim=None):
    """(w_ST, w_EL) floor with Belief on z — fixed limits from the prior pdf peak."""
    from matplotlib.ticker import MaxNLocator

    dlo1, dhi1, dlo2, dhi2, _, _ = CH5_VIEW_BOUNDS
    pk = str(prior_kind).lower()
    if z_lim is None:
        z_lim = ch5_prior_w12_z_lim(pk)
    fs = float(_g("AXIS_LABEL_SIZE")) * float(_g("CH3_LIK_3D_AXIS_LABEL_SCALE"))
    ax3d.set_xlim(dlo1, dhi1)
    ax3d.set_ylim(dlo2, dhi2)
    ax3d.set_zlim(float(z_lim[0]), float(z_lim[1]))
    ax3d.set_xlabel(r"$w_{\mathrm{ST}}$", fontsize=fs, labelpad=10)
    ax3d.set_ylabel(r"$w_{\mathrm{EL}}$", fontsize=fs, labelpad=10)
    ax3d.set_zlabel("Belief", fontsize=fs, labelpad=10)
    ax3d.zaxis.set_major_locator(MaxNLocator(nbins=5))
    ax3d.tick_params(axis="both", which="major", labelsize=float(_g("FONT_SIZE")))


def _style_ax3d_mini(ax3d, *, prior_kind="gaussian", z_lim=None):
    _ch5_style_belief_w12_ax3d(ax3d, prior_kind=prior_kind, z_lim=z_lim)
    ax3d.tick_params(labelsize=6)
    _g("ch4_lik_ct_view_init")(ax3d)


def _ch5_map_z_on_surface(W1, W2, density, ws, we, *, squish_u=0.0, prior_kind="gaussian", z_lim=None):
    pk = str(prior_kind).lower()
    if z_lim is None:
        z_lim = ch5_prior_w12_z_lim(pk)
    z = ch5_clip_belief_height(density, prior_kind=pk, z_lim=z_lim)
    u = float(np.clip(squish_u, 0.0, 1.0))
    z_show = z * (1.0 - u)
    return float(_g("_ch3_lik_w12_z_at")(W1, W2, z_show, float(ws), float(we)))


def _ch5_draw_map_peak_marker(
    fig, ax3d, ws, we, z_pt, *, here_annotation=True, label="most plausible line",
):
    """Ch4-style MAP marker + arrow on the 3D belief landscape (47+ wording)."""
    mx, my = float(ws), float(we)
    z_lo, z_hi = ax3d.get_zlim()
    z_span = max(float(z_hi) - float(z_lo), 1e-12)
    z_mark = float(z_pt) + float(_g("CH3_LIK_MARKER_Z_BUMP_FRAC")) * z_span
    mcol = _g("FAIL_COLOR")
    ax3d.scatter(
        [mx], [my], [z_mark],
        color=mcol, edgecolors="white", linewidths=2.0,
        s=260.0, depthshade=False, zorder=30,
    )
    if here_annotation:
        _g("_ch3_lik_w12_here_annotation")(
            fig, ax3d, mx, my, z_mark,
            label=str(label),
            color=mcol, text_color="white", edgecolor="white",
        )


def _draw_roster_ax(ax, pack, *, show_line=True, alpha=1.0, title=None):
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    xl, yl = ch5_plot_limits(pack["key"])
    if show_line:
        _ch5_draw_data_panel(
            ax, study, exam, y, xl=xl, yl=yl,
            show_colormap=False, show_threshold=False,
        )
    else:
        _g("draw_dataset")(ax, study, exam, y)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        _g("finalize_style_legend_tex")(ax)
    if title:
        ax.text(0.03, 0.97, title, transform=ax.transAxes, va="top", ha="left", fontsize=9, color="#333", alpha=alpha)


def _ch5_subset_pack(pack, n_show):
    order = pack["order"]
    if n_show is None:
        n_show = len(order)
    n_show = int(min(max(0, n_show), len(order)))
    idx = np.asarray(order[:n_show], dtype=int)
    if idx.size == 0:
        return pack["study"][:0], pack["exam"][:0], pack["y"][:0], 0
    return pack["study"][idx], pack["exam"][idx], pack["y"][idx], n_show


def _ch5_map_weights(pack, *, prior_kind="gaussian"):
    """Continuous MAP (w_ST, w_EL, b) — same optimizer as the 3D voxel credible clips."""
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    cont = ch5_posterior_map_continuous(
        study, exam, y,
        nll_fn=_g("_ch3_nll_sum_on_flat_grid"),
        nll_grad_fn=_g("_ch3_nll_sum_grad_at_point"),
        prior_kind=str(prior_kind).lower(),
        x0=pack["display_w"],
    )
    return float(cont["ws"]), float(cont["we"]), float(cont["bb"])


def _ch5_w12_surface_for_n(pack, n_show, *, prior_kind="gaussian"):
    study, exam, y, _ = _ch5_subset_pack(pack, n_show)
    return ch5_posterior_w12_surface(
        study, exam, y, prior_kind=prior_kind, nll_fn=_g("_ch3_nll_sum_on_flat_grid"),
    )


def _draw_posterior_surface(
    ax3d,
    surface,
    *,
    colored=True,
    mono=False,
    squish_u=0.0,
    credible_mass=None,
    prior_red=False,
    fig=None,
    show_map_marker=False,
    map_ws=None,
    map_we=None,
    prior_kind="gaussian",
):
    W1, W2, density = surface["W1"], surface["W2"], surface["density"]
    pk = str(surface.get("prior_kind", prior_kind)).lower()
    z_lim = surface.get("z_lim", ch5_prior_w12_z_lim(pk))
    _style_ax3d_mini(ax3d, prior_kind=pk, z_lim=z_lim)
    _draw_density_surface(
        ax3d, W1, W2, density, colored=colored, mono=mono,
        squish_u=squish_u, prior_red=prior_red, prior_kind=pk, z_lim=z_lim,
    )
    if credible_mass is not None:
        log_post = surface.get("log_post")
        if log_post is not None:
            hpd_density = ch5_posterior_grid_weights(log_post)
        else:
            hpd_density = density
        ch5_hpd_region_on_floor(ax3d, W1, W2, hpd_density, mass=float(credible_mass))
    if show_map_marker and fig is not None:
        mws = surface.get("ws") if map_ws is None else float(map_ws)
        mwe = surface.get("we") if map_we is None else float(map_we)
        if mws is not None and mwe is not None:
            z_pt = _ch5_map_z_on_surface(
                W1, W2, density, mws, mwe, squish_u=squish_u, prior_kind=pk, z_lim=z_lim,
            )
            _ch5_draw_map_peak_marker(fig, ax3d, mws, mwe, z_pt)


def _draw_posterior_ax(ax3d, pack, *, colored=True, mono=False, squish_u=0.0, credible_mass=None):
    _draw_posterior_surface(
        ax3d,
        {"W1": pack["W1"], "W2": pack["W2"], "density": pack["density"]},
        colored=colored, mono=mono, squish_u=squish_u, credible_mass=credible_mass,
    )


def _draw_duo_data_panel(
    ax_data, pack, *, n_show=None, title=False, ws=None, we=None, bb=None,
    show_threshold=False, extra_thresholds=None, threshold_legend=True,
    threshold_linewidth=None, threshold_linestyle="--", threshold_color="grey",
):
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    order = pack["order"]
    key = pack["key"]
    if n_show is None:
        n_show = len(y)
    n_show = int(min(max(0, n_show), len(order)))
    if ws is None or we is None or bb is None:
        ws, we, bb = pack["ws"], pack["we"], pack["bb"]
    else:
        ws, we, bb = float(ws), float(we), float(bb)
    xl, yl = ch5_plot_limits(key)
    mask = np.zeros(len(y), dtype=bool)
    for j in order[:n_show]:
        mask[int(j)] = True
    _ch5_draw_data_panel(
        ax_data, study, exam, y,
        xl=xl, yl=yl, mask=mask, ws=ws, we=we, bb=bb,
        show_threshold=show_threshold,
        extra_thresholds=extra_thresholds,
        threshold_legend=threshold_legend,
        threshold_linewidth=threshold_linewidth,
        threshold_linestyle=threshold_linestyle,
        threshold_color=threshold_color,
    )
    if title and pack["meta"].get("title"):
        ax_data.text(
            0.03, 0.97, pack["meta"]["title"], transform=ax_data.transAxes,
            va="top", ha="left", fontsize=9, color="#333",
        )
    return ws, we, bb


def _draw_duo_knobs(fig, ax_data, axes_k, ws, we, bb):
    knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
    _g("ch3_draw_knob_row")(
        fig, axes_k, ws, we, bb, "st", knob_rgbs, canvas_sides,
        rot_strip_deg=0.0, strip_scale=1.0,
        knob_rots=_g("ch3_k1_knob_rots_at")(ws, we, bb), knob_scales=[1.0, 1.0, 1.0], ax_data=ax_data,
    )
    _g("_ch3_align_knob_axes_under_data")(fig, ax_data, axes_k)
    _g("ch3_layout_knob_axes_like_bridge_end")(fig, ax_data, axes_k)


def _ch5_place_knobs_in_rects(fig, ax_data, data_r, knob_rs, ws, we, bb):
    """Place w_ST / w_EL / b labeled knobs (ch4 pack), not numbered ch3 dials."""
    targets = _g("_ch4_02b_knob_target_rects")(data_r, knob_rs)
    axes_k = tuple(fig.add_axes(r) for r in targets)
    for ax in axes_k:
        ax.axis("off")
    _draw_duo_knobs(fig, ax_data, axes_k, ws, we, bb)


def _ch5_place_knobs_flyin(fig, ax_data, data_r, knob_rs, ws, we, bb, *, grow_u=1.0):
    """Fly w_ST / w_EL / b labeled knobs from legend params into their slots."""
    grow_u = float(np.clip(float(grow_u), 0.0, 1.0))
    if grow_u <= 1e-4:
        return
    if grow_u >= 1.0 - 1e-6:
        _ch5_place_knobs_in_rects(fig, ax_data, data_r, knob_rs, ws, we, bb)
        return
    _g("ch4_ensure_labeled_knob_pngs")()
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


def _fig_to_plot(fig):
    return _g("fig_to_image")(fig, dpi=_g("CH3_ANIM_DPI"))


def _ch5_plot_2d_dpi() -> float:
    """Match ch4_02b animation DPI when available."""
    try:
        return float(_g("_ch4_02b_anim_dpi")())
    except (KeyError, TypeError):
        return float(_g("CH3_ANIM_DPI"))


def _fig_to_plot_2d(fig):
    """Rasterize a standalone 2D panel at chapter-4 animation resolution."""
    return _g("fig_to_image")(fig, dpi=_ch5_plot_2d_dpi())


def _ch5_ch4_wide_figure():
    """Figure + axes matching ch4_02b wide 2D layout (CH4_DUO_FIGSIZE)."""
    wide = _G.get("CH4_02B_WIDE_DATA", CH5_CH4_WIDE_DATA)
    fig = plt.figure(figsize=CH4_DUO_FIGSIZE)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes(wide)
    return fig, ax


def _frame_plot_2d_dataset(
    key,
    *,
    n_show=None,
    ws=None,
    we=None,
    bb=None,
    show_threshold=False,
    extra_thresholds=None,
    threshold_legend=True,
    threshold_linewidth=None,
    threshold_linestyle="--",
    threshold_color="grey",
    title=False,
):
    """Standalone 2D dataset panel — ch4 wide layout + animation DPI."""
    pack = _PACKS[key]
    fig, ax = _ch5_ch4_wide_figure()
    _draw_duo_data_panel(
        ax, pack, n_show=n_show, ws=ws, we=we, bb=bb,
        show_threshold=show_threshold, extra_thresholds=extra_thresholds,
        threshold_legend=threshold_legend, threshold_linewidth=threshold_linewidth,
        threshold_linestyle=threshold_linestyle, threshold_color=threshold_color,
        title=title,
    )
    return _fig_to_plot_2d(fig)


def _ch5_bias_sweep_bb_values(map_bb, *, delta=None):
    """Smooth bias sweep: center → +δ → center → −δ → center."""
    delta = float(CH5_BIAS_SWEEP_DELTA if delta is None else delta)
    n = _draft_range(CH5_BIAS_SWEEP_N, 6)
    vals = []
    for sign in (+1.0, -1.0):
        vals.extend(np.linspace(map_bb, map_bb + sign * delta, n, endpoint=True))
        vals.extend(np.linspace(map_bb + sign * delta, map_bb, n, endpoint=True))
    return vals


def _build_dataset_2d_map_story(clip_id):
    """D1 sequential points → MAP line → bias sweep; erase; D4 + MAP + iso-posterior lines."""
    frames = []
    pack1 = _PACKS["D1"]
    order1 = pack1["order"]
    n1 = len(order1)
    map_ws, map_we, map_bb = _ch5_map_weights(pack1)

    def add(img, *, hold=CH5_N_SEQ_HOLD):
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], hold))

    add(_frame_plot_2d_dataset("D1", n_show=0), hold=CH5_N_HOLD)
    for n in range(1, n1 + 1):
        add(_frame_plot_2d_dataset("D1", n_show=n))
    add(
        _frame_plot_2d_dataset(
            "D1", n_show=n1, ws=map_ws, we=map_we, bb=map_bb, show_threshold=True,
        ),
        hold=CH5_N_HOLD,
    )
    for bb in _ch5_bias_sweep_bb_values(map_bb):
        frames.append(_finish(
            _frame_plot_2d_dataset(
                "D1", n_show=n1, ws=map_ws, we=map_we, bb=float(bb), show_threshold=True,
            ),
            clip_id,
        ))
    frames.extend(_hold(frames[-1], CH5_N_HOLD))

    add(_frame_plot_2d_dataset("D4", n_show=0), hold=CH5_N_HOLD)

    pack4 = _PACKS["D4"]
    order4 = pack4["order"]
    n4 = len(order4)
    map4_ws, map4_we, map4_bb = _ch5_map_weights(pack4)
    for n in range(1, n4 + 1):
        add(_frame_plot_2d_dataset("D4", n_show=n))
    add(
        _frame_plot_2d_dataset(
            "D4", n_show=n4, ws=map4_ws, we=map4_we, bb=map4_bb, show_threshold=True,
        ),
        hold=CH5_N_HOLD,
    )
    for k in range(1, len(CH5_D4_ALT_LINES) + 1):
        extra = [
            (*CH5_D4_ALT_LINES[i], CH5_ALT_LINE_COLORS[i])
            for i in range(k)
        ]
        add(
            _frame_plot_2d_dataset(
                "D4", n_show=n4, ws=map4_ws, we=map4_we, bb=map4_bb,
                show_threshold=True, extra_thresholds=extra,
            ),
            hold=CH5_N_HOLD,
        )
    frames.extend(_hold(frames[-1]))
    return frames


def _ch5_map_weights_arrays(study, exam, y, *, prior_kind="gaussian", x0=(0.5, -0.5, 0.0)):
    cont = ch5_posterior_map_continuous(
        study, exam, y,
        nll_fn=_g("_ch3_nll_sum_on_flat_grid"),
        nll_grad_fn=_G.get("_ch3_nll_sum_grad_at_point"),
        prior_kind=str(prior_kind).lower(),
        x0=tuple(float(v) for v in x0),
    )
    return float(cont["ws"]), float(cont["we"]), float(cont["bb"])


def _ch5_d1_variants():
    pack = _PACKS["D1"]
    study0 = np.asarray(pack["study"], dtype=np.float64)
    exam0 = np.asarray(pack["exam"], dtype=np.float64)
    y0 = np.asarray(pack["y"], dtype=np.int64)
    out = []
    for dx, dy in CH5_D1_VARIANT_SHIFTS[:CH5_INTRO_N_VARIANTS]:
        st, el, yy = [], [], []
        for s, e, yi in zip(study0, exam0, y0):
            ns = float(np.clip(float(s) + float(dx), 0.8, 6.5))
            ne = float(np.clip(float(e) + float(dy), 0.8, 6.5))
            label = 1 if ns > ne else 0
            st.append(ns)
            el.append(ne)
            yy.append(label)
        out.append((
            np.asarray(st, dtype=np.float64),
            np.asarray(el, dtype=np.float64),
            np.asarray(yy, dtype=np.int64),
        ))
    return out


def _frame_plot_2d_points(
    study,
    exam,
    y,
    *,
    mask=None,
    ws=None,
    we=None,
    bb=None,
    show_threshold=False,
    after_draw=None,
    xl=None,
    yl=None,
):
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if xl is None or yl is None:
        xl0, yl0 = ch5_plot_limits("D1")
        xl = xl0 if xl is None else xl
        yl = yl0 if yl is None else yl
    fig, ax = _ch5_ch4_wide_figure()
    _ch5_draw_data_panel(
        ax, study, exam, y, xl=xl, yl=yl, mask=mask,
        ws=ws, we=we, bb=bb, show_threshold=show_threshold,
        after_draw=after_draw,
    )
    return _fig_to_plot_2d(fig)


def _ch5_intro_ch5_duo_layout():
    global _CH5_INTRO_DUO_RECTS
    if _CH5_INTRO_DUO_RECTS is not None:
        return _CH5_INTRO_DUO_RECTS

    def _ax_rect(ax):
        bb = ax.get_position()
        return (float(bb.x0), float(bb.y0), float(bb.width), float(bb.height))

    fig, ax_data, ax_right, axes_k = _g("ch6_figure_duo")()
    fig.canvas.draw()
    _CH5_INTRO_DUO_RECTS = (
        _ax_rect(ax_data),
        tuple(_ax_rect(ax) for ax in axes_k),
        _ax_rect(ax_right),
    )
    plt.close(fig)
    return _CH5_INTRO_DUO_RECTS


def _ch5_sequential_bars_duo_layout():
    """ch5 duo rects after the same knob alignment as ``_frame_sequential_bars``."""
    global _CH5_SEQ_BARS_DUO_RECTS
    if _CH5_SEQ_BARS_DUO_RECTS is not None:
        return _CH5_SEQ_BARS_DUO_RECTS

    def _ax_rect(ax):
        bb = ax.get_position()
        return (float(bb.x0), float(bb.y0), float(bb.width), float(bb.height))

    fig, ax_data, ax_right, axes_k = _g("ch6_figure_duo")()
    ws, we, bb = CH5_THREE_LINES[1]
    knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
    _g("ch3_draw_knob_row")(
        fig, axes_k, ws, we, bb, "all", knob_rgbs, canvas_sides,
        rot_strip_deg=0.0, strip_scale=1.0,
        knob_rots=_g("ch3_k1_knob_rots_at")(ws, we, bb), knob_scales=[1.0, 1.0, 1.0],
        ax_data=ax_data,
    )
    _g("_ch3_align_knob_axes_under_data")(fig, ax_data, axes_k)
    _g("ch3_layout_knob_axes_like_bridge_end")(fig, ax_data, axes_k)
    _g("ch6_duo_right_panel_extent")(fig, ax_data, ax_right, axes_k)
    fig.canvas.draw()
    _CH5_SEQ_BARS_DUO_RECTS = (
        _ax_rect(ax_data),
        tuple(_ax_rect(ax) for ax in axes_k),
        _ax_rect(ax_right),
    )
    plt.close(fig)
    return _CH5_SEQ_BARS_DUO_RECTS


def _frame_ch5_intro_duo_plot(
    layout_u,
    knob_grow_u,
    bar_u,
    *,
    study=None,
    exam=None,
    y=None,
    mask=None,
    n_lines=0,
    line_weights=None,
    winner_idx=None,
    n_bars=0,
):
    """Wide 2D → ch5 duo (2D + knobs + histogram), ch4_02b-style rect morph."""
    lu = float(np.clip(float(layout_u), 0.0, 1.0))
    kg = float(np.clip(float(knob_grow_u), 0.0, 1.0))
    bu = float(np.clip(float(bar_u), 0.0, 1.0))

    wide_data, wide_knobs = _g("_ch4_02b_wide_layout")()
    duo_data, duo_knobs, duo_bar = _ch5_intro_ch5_duo_layout()
    smooth = _g("ch3_knob_smoothstep")
    su = float(smooth(lu))
    data_r = _g("_ch4_02b_lerp_rect")(su, wide_data, duo_data)
    knob_rs = tuple(_g("_ch4_02b_lerp_rect")(su, wide_knobs[i], duo_knobs[i]) for i in range(3))
    bar_r = _g("_ch4_02b_lerp_rect")(su, duo_bar, duo_bar)

    pack = _PACKS["D1"]
    if study is None:
        study, exam, y = pack["study"], pack["exam"], pack["y"]
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if mask is None:
        mask = np.zeros(len(y), dtype=bool)
    xl, yl = ch5_plot_limits("D1")

    def _lines(ax, _xl, _yl):
        if winner_idx is not None:
            _draw_hypothesis_lines_on_ax(
                ax, _xl, _yl, CH5_FIVE_LINES, CH5_FIVE_LINE_COLORS,
                winner_idx=winner_idx,
            )
        elif n_lines > 0:
            _draw_hypothesis_lines_on_ax(
                ax, _xl, _yl, CH5_FIVE_LINES, CH5_FIVE_LINE_COLORS,
                n_lines=n_lines, line_weights=line_weights,
            )

    ws, we, bb = CH5_THREE_LINES[1]
    fig = plt.figure(figsize=CH4_DUO_FIGSIZE)
    fig.patch.set_facecolor("white")
    ax_data = fig.add_axes(data_r)
    _ch5_draw_data_panel(
        ax_data, study, exam, y, xl=xl, yl=yl, mask=mask, after_draw=_lines,
    )
    _ch5_place_knobs_flyin(fig, ax_data, data_r, knob_rs, ws, we, bb, grow_u=kg)
    if bu > 1e-3:
        ax_bar = fig.add_axes(bar_r)
        ax_bar.patch.set_alpha(bu)
        for spine in ax_bar.spines.values():
            spine.set_alpha(bu)
        _draw_sequential_bars_on_ax(ax_bar, n_bars=n_bars)
        ax_bar.patch.set_alpha(bu)
    return _fig_to_plot_2d(fig)


def _ch5_01_opening_frame(compose_id: str = "ch5_01"):
    """Exact composed frame matching the first beat of ch5_01_sequential_bars."""
    empty = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)
    plot = _cached_frame_sequential_bars(n_lines=0, n_bars=0, n_show=0)
    right_blocks = _ch5_exact_values_blocks(empty, empty, empty_y)
    right_prog = _ch5_seq_right_progress(n_lines=0, n_prior_vals=0)
    return _ch5_finish_bars(
        plot,
        compose_id,
        right_blocks=right_blocks,
        progress_override={"right": right_prog},
    )


def _build_intro_to_01_story(clip_id, *, compose_id: str = "ch5_01"):
    """Preamble: D1 MAP variants → five-line race → layout morph into ch5_01 opening."""
    frames = []
    pack = _PACKS["D1"]
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    point_order = _ch5_d1_sequential_point_order()
    n_pts = len(point_order)
    map_ws, map_we, map_bb = _ch5_map_weights(pack)
    variants = _ch5_d1_variants()
    hold = _draft_range(6, 2)
    hold_short = _draft_range(4, 2)
    plot_rect = _g("CH4_LIK_PLOT_START_RECT")
    empty_mask = np.zeros(len(y), dtype=bool)
    global _CH5_INTRO_DUO_RECTS
    _CH5_INTRO_DUO_RECTS = None

    def add(img, n=None):
        frames.extend(_hold(img, hold_short if n is None else n))

    def add_comp(img, *, layout_u=1.0, panel_u=1.0, title_u=1.0, write_u=1.0, plot_start=None):
        empty = np.array([], dtype=np.float64)
        empty_y = np.array([], dtype=np.int64)
        right_blocks = _ch5_exact_values_blocks(empty, empty, empty_y)
        right_prog = _ch5_seq_right_progress(n_lines=0, n_prior_vals=0)
        kw = dict(
            layout_u=layout_u,
            panel_u=panel_u,
            title_write_progress=title_u,
            write_progress=write_u,
        )
        if plot_start is not None:
            kw["plot_start_rect"] = plot_start
        frames.append(_ch5_finish_bars(
            img, compose_id,
            right_blocks=right_blocks,
            progress_override={"right": right_prog},
            **kw,
        ))

    # 1 — D1 + MAP threshold (full canvas).
    add(_frame_plot_2d_dataset(
        "D1", n_show=n_pts, ws=map_ws, we=map_we, bb=map_bb, show_threshold=True,
    ), hold)

    # 2 — four similar regenerated datasets + MAP each.
    for vst, vel, vy in variants:
        vws, vwe, vbb = _ch5_map_weights_arrays(vst, vel, vy)
        add(_frame_plot_2d_points(
            vst, vel, vy, mask=np.ones(len(vy), dtype=bool),
            ws=vws, we=vwe, bb=vbb, show_threshold=True,
        ), hold_short)

    # 3 — original D1 + MAP.
    add(_frame_plot_2d_dataset(
        "D1", n_show=n_pts, ws=map_ws, we=map_we, bb=map_bb, show_threshold=True,
    ), hold)

    # 4 — erase data; five competing lines (3 from ch5_01 + 2 extra).
    add(_frame_plot_2d_points(
        study, exam, y, mask=empty_mask,
        after_draw=lambda ax, xl, yl: _draw_hypothesis_lines_on_ax(
            ax, xl, yl, CH5_FIVE_LINES, CH5_FIVE_LINE_COLORS, n_lines=5,
        ),
    ), hold)

    # 5 — points appear; line thickness tracks discrete posterior mass.
    winner = 0
    for n in range(1, n_pts + 1):
        mask = np.zeros(len(y), dtype=bool)
        for j in point_order[:n]:
            mask[int(j)] = True
        sn = study[mask]
        en = exam[mask]
        yn = y[mask]
        belief = _ch5_five_line_belief(sn, en, yn, stage="posterior")
        winner = int(np.argmax(belief))
        add(_frame_plot_2d_points(
            study, exam, y, mask=mask,
            after_draw=lambda ax, xl, yl, b=belief: _draw_hypothesis_lines_on_ax(
                ax, xl, yl, CH5_FIVE_LINES, CH5_FIVE_LINE_COLORS,
                n_lines=5, line_weights=b,
            ),
        ), 1 if _CH3_DRAFT else 2)

    # 6 — survivor → single dashed threshold.
    add(_frame_plot_2d_points(
        study, exam, y, mask=np.ones(len(y), dtype=bool),
        after_draw=lambda ax, xl, yl, w=winner: _draw_hypothesis_lines_on_ax(
            ax, xl, yl, CH5_FIVE_LINES, CH5_FIVE_LINE_COLORS, winner_idx=w,
        ),
    ), hold)

    # 7 — erase plot contents (empty full canvas).
    add(_frame_plot_2d_points(study, exam, y, mask=empty_mask), hold_short)

    # 8 — knobs fly in; plot morphs to duo-left + empty histogram.
    n_duo = _draft_range(CH5_INTRO_N_DUO_MORPH, 6)
    for tv in np.linspace(0.0, 1.0, n_duo, endpoint=True):
        u = float(_g("ch3_knob_smoothstep")(float(tv)))
        frames.append(_frame_ch5_intro_duo_plot(u, u, u, mask=empty_mask, n_lines=0, n_bars=0))
    add(frames[-1], hold_short)

    plot0 = frames[-1]

    # 9 — tutorial shell: full-width plot shrinks into left slot.
    n_layout = _draft_range(CH5_INTRO_N_LAYOUT_MORPH, 6)
    for tv in np.linspace(0.0, 1.0, n_layout, endpoint=True):
        u = float(_g("ch3_knob_smoothstep")(float(tv)))
        add_comp(plot0, layout_u=u, panel_u=0.0, title_u=0.0, write_u=0.0, plot_start=plot_rect)

    # 10 — text rail backgrounds + “Exact Values” title.
    n_panel = _draft_range(CH5_INTRO_N_PANEL_MORPH, 5)
    for tv in np.linspace(0.0, 1.0, n_panel, endpoint=True):
        u = float(_g("ch3_knob_smoothstep")(float(tv)))
        add_comp(plot0, layout_u=1.0, panel_u=u, title_u=u, write_u=1.0, plot_start=plot_rect)

    # 11 — hold on exact ch5_01 opening frame.
    opening = _ch5_01_opening_frame(compose_id)
    add(opening, hold)
    frames.extend(_hold(opening))
    return frames


def _ch5_weight3d_duo_layout():
    global _CH5_WEIGHT3D_DUO_RECTS
    if _CH5_WEIGHT3D_DUO_RECTS is not None:
        return _CH5_WEIGHT3D_DUO_RECTS

    def _ax_rect(ax):
        bb = ax.get_position()
        return (float(bb.x0), float(bb.y0), float(bb.width), float(bb.height))

    fig, ax_data, ax3d, axes_k = _g("ch4_figure_duo_weight3d")()
    fig.canvas.draw()
    _CH5_WEIGHT3D_DUO_RECTS = (
        _ax_rect(ax_data),
        tuple(_ax_rect(ax) for ax in axes_k),
        _ax_rect(ax3d),
    )
    plt.close(fig)
    return _CH5_WEIGHT3D_DUO_RECTS


def _frame_ch5_morph_hist_to_weight3d(
    morph_u,
    *,
    ws=0.0,
    we=0.0,
    bb=0.0,
    study=None,
    exam=None,
    y=None,
    mask=None,
    n_bars=0,
    bar_heights=None,
    bar_label_mode=None,
    n_lines=0,
    line_weights=None,
    empty_3d=False,
    prior_kind="uniform",
    elev=None,
    azim=None,
):
    """Morph ch5 duo (histogram) → ch4 duo weight3d; optional empty 3-D panel."""
    u = float(np.clip(float(morph_u), 0.0, 1.0))
    su = float(_g("ch3_knob_smoothstep")(u))
    duo_data, duo_knobs, duo_bar = _ch5_sequential_bars_duo_layout()
    w3d_data, w3d_knobs, w3d_3d = _ch5_weight3d_duo_layout()
    data_r = _g("_ch4_02b_lerp_rect")(su, duo_data, w3d_data)
    knob_rs = tuple(_g("_ch4_02b_lerp_rect")(su, duo_knobs[i], w3d_knobs[i]) for i in range(3))
    right_r = _g("_ch4_02b_lerp_rect")(su, duo_bar, w3d_3d)

    pack = _PACKS["D1"]
    if study is None:
        study, exam, y = pack["study"], pack["exam"], pack["y"]
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if mask is None:
        mask = np.zeros(len(y), dtype=bool)
    xl, yl = ch5_plot_limits("D1")

    def _lines(ax, _xl, _yl):
        if n_lines > 0:
            _draw_three_lines_on_ax(
                ax, _xl, _yl, n_lines=n_lines, line_weights=line_weights,
            )

    fig = plt.figure(figsize=CH4_DUO_FIGSIZE)
    fig.patch.set_facecolor("white")
    ax_data = fig.add_axes(data_r)
    _ch5_draw_data_panel(
        ax_data, study, exam, y, xl=xl, yl=yl, mask=mask,
        after_draw=_lines, show_colormap=False, show_threshold=False,
    )
    _ch5_place_knobs_in_rects(fig, ax_data, data_r, knob_rs, ws, we, bb)

    bar_alpha = max(0.0, 1.0 - su)
    if bar_alpha > 1e-3 and n_bars > 0:
        ax_bar = fig.add_axes(right_r)
        ax_bar.patch.set_alpha(bar_alpha)
        for spine in ax_bar.spines.values():
            spine.set_alpha(bar_alpha)
        _draw_sequential_bars_on_ax(
            ax_bar,
            n_bars=n_bars,
            heights=bar_heights,
            label_mode=bar_label_mode,
            n_labels=n_bars if bar_label_mode in ("posterior", "multiply") else 0,
            bar_ymax=CH5_SEQ_BAR_YMAX_POST if bar_label_mode == "posterior" else None,
        )

    if empty_3d and su > 1e-3:
        ax3d = fig.add_axes(right_r, projection="3d")
        _ch5_style_belief_w12_ax3d(
            ax3d, prior_kind=prior_kind, z_lim=ch5_prior_w12_z_lim(prior_kind, scope="prior"),
        )
        el = float(_g("CH3_LIK_W12_ELEV_W1") if elev is None else elev)
        az = float(_g("CH3_LIK_W12_AZIM_W1") if azim is None else azim)
        ax3d.view_init(elev=el, azim=az)
        ax3d.patch.set_alpha(su)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.pane.fill = False
            axis.pane.set_alpha(su * 0.35)
    return _fig_to_plot(fig)


def _ch5_compose_bars_frame(
    plot_img,
    compose_id: str,
    *,
    layout_u=1.0,
    panel_u=1.0,
    title_u=1.0,
    write_u=1.0,
    plot_start=None,
    bottom_blocks=None,
    bottom_prog=None,
    corner_blocks=None,
    corner_u=0.0,
    right_prog=None,
    right_blocks=None,
    shell_cache_key: str | None = None,
    rails_cache_key: str | None = None,
    n_show=0,
):
    empty = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)
    if right_blocks is None:
        right_blocks = _ch5_exact_values_blocks(empty, empty, empty_y)
    if right_prog is None:
        right_prog = _ch5_seq_right_progress(n_lines=0, n_prior_vals=0, full_values=True)
    po = {"right": right_prog}
    if bottom_blocks is not None and bottom_prog is not None:
        po["bottom"] = ch4_bottom_per_block_progress(bottom_blocks, bottom_prog)
    if corner_blocks is not None:
        style = _ch5_seq_composer("balanced").handwrite_style()
        po["corner"] = ch4_blocks_write_from_slot(corner_blocks, 0, float(corner_u), style=style)
    kw = dict(
        layout_u=float(layout_u),
        panel_u=float(panel_u),
        title_write_progress=float(title_u),
        write_progress=float(write_u),
        progress_override=po,
    )
    if plot_start is not None:
        kw["plot_start_rect"] = plot_start
    if bottom_blocks is not None:
        kw["bottom_blocks"] = bottom_blocks
    if corner_blocks is not None:
        kw["corner_blocks"] = corner_blocks
        kw["show_corner"] = True
    return _ch5_finish_bars(
        plot_img, compose_id, right_blocks=right_blocks,
        shell_cache_key=shell_cache_key,
        rails_cache_key=rails_cache_key,
        **kw,
    )


def _ch5_compose_plot_only(plot_img):
    """Full-bleed plot on the export canvas — ch4 duo split-screen, no tutorial rails."""
    return _g("compose_tutorial")(
        plot_img,
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


def _ch5_finish_duo_export(plot_img, clip_id: str, *, show_hessian: bool = False):
    """Plot-only clips: full-bleed duo on the chapter-4 export canvas (no letterboxing)."""
    del show_hessian  # Hessian is drawn in-figure on the side panel now.
    if _use_text(clip_id):
        return _finish(plot_img, clip_id)
    return _ch5_compose_plot_only(plot_img)


def _ch5_smoothstep(u: float) -> float:
    t = float(np.clip(u, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _ch5_apply_hessian_side_slide(fig, ax_data, axes_k, ax3d, *, slide_u: float):
    """
    Slide 2D+knobs off the left; translate the 3D axes left keeping its on-screen
    inch size (aspect preserved); return a right-side axes for the Hessian panel.
    """
    u = _ch5_smoothstep(slide_u)
    if u <= 1e-6:
        return None

    d = ax_data.get_position()
    r = ax3d.get_position()
    fig_w, fig_h = fig.get_size_inches()
    # Push the 2D column fully off-canvas to the left.
    dx_off = -(float(d.x0) + float(d.width) + 0.04) * u
    ax_data.set_position([
        float(d.x0) + dx_off, float(d.y0), float(d.width), float(d.height),
    ])
    for axk in axes_k:
        p = axk.get_position()
        axk.set_position([
            float(p.x0) + dx_off, float(p.y0), float(p.width), float(p.height),
        ])

    # Keep the 3D panel's physical inch size (no stretch), park it on the left.
    w_in = float(r.width) * float(fig_w)
    h_in = float(r.height) * float(fig_h)
    w = w_in / float(fig_w)
    h = h_in / float(fig_h)
    target_x0 = 0.025
    # Vertically center relative to original 3D vertical mid.
    mid_y = float(r.y0) + 0.5 * float(r.height)
    target_y0 = mid_y - 0.5 * h
    x0 = float(r.x0) + (target_x0 - float(r.x0)) * u
    y0 = float(r.y0) + (target_y0 - float(r.y0)) * u
    ax3d.set_position([x0, y0, w, h])

    gap = 0.04
    hx0 = x0 + w + gap
    hw = max(0.0, 0.98 - hx0)
    if hw < 0.12:
        return None
    panel_u = float(np.clip((u - 0.30) / 0.70, 0.0, 1.0))
    if panel_u <= 1e-6:
        return None
    # Tall text panel (full plot height) so fraction gaps stay readable after fit.
    hy0 = 0.06
    hh = 0.88
    ax_h = fig.add_axes([hx0, hy0, hw, hh])
    ax_h.set_axis_off()
    ax_h.set_facecolor("none")
    for spine in ax_h.spines.values():
        spine.set_visible(False)
    return ax_h, panel_u


def _ch5_draw_hessian_side_panel(ax, *, cell_colors=None, write_u=1.0):
    """Centered handwritten Hessian matrix in a side axes."""
    from ch4_layout import CH4_HERE_BLOCK_FS

    style = CH4_COMPOSER.handwrite_style()
    block = ch4_formula_hessian_matrix_block(
        colored_cells=False,
        weight=1.0,
        text_x_frac=0.5,
        matrix_max_frac=0.90,
        align="center",
        block_fs=float(CH4_HERE_BLOCK_FS) * 1.15,
        top_pad_pt=0.0,
        text_y_inset_pt=0.0,
        matrix_x_shift_pt=0.0,
        matrix_y_shift_pt=0.0,
        handwrite_matrix={
            "align": "center",
            "cell_fs_scale": 0.86,
            "fill_height_frac": 0.0,
            "row_gap_pt": 18.0,
            "col_gap_pt": 20.0,
            "cell_frac_gap_frac": 1.05,
            "cell_frac_pad_frac": 0.24,
            "cell_frac_scale": 0.86,
            "bracket_y_shift_pt": 0.0,
            "matrix_cell_y_shift_pt": 0.0,
            "cell_colors": cell_colors,
        },
    )
    # Generous top inset so the block sits mid-panel after fit.
    draw_handwrite_matrix_in_cell(
        ax, block,
        style=style,
        block_fs=float(block.get("block_fs", CH4_HERE_BLOCK_FS)),
        align="center",
        line_progress={0: float(np.clip(write_u, 0.0, 1.0))},
        text_x_frac=0.5,
        text_y_inset_pt=36.0,
    )


def _ch5_01_closing_state():
    """Plot/text state matching the last beat of ch5_01 sequential_bars."""
    pack = _PACKS["D1"]
    point_order = _ch5_d1_sequential_point_order()
    n_pts = len(point_order)
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    idxs = [int(j) for j in point_order]
    sn = study[idxs]
    en = exam[idxs]
    yn = y[idxs]
    post = _ch5_three_line_belief(sn, en, yn, stage="posterior")
    plot_kw = dict(
        n_lines=3, n_bars=3, n_show=n_pts,
        bar_heights=post, bar_label_mode="posterior",
        line_weights=post,
    )
    plot = _cached_frame_sequential_bars(**plot_kw)
    post_frac_bottom = _ch5_relevant_equations_blocks(bayes_frac=CH5_SEQ_BAYES_FRAC_TEX)
    corner_blocks = _g("ch4_cached_notation_corner_blocks")()
    full_bottom_prog = _ch5_seq_bottom_prog(bayes=1.0, lik=1.0, prob=1.0)
    right_prog = _ch5_seq_right_progress(n_lines=3, n_prior_vals=0, full_values=True)
    right_blocks = _ch5_exact_values_blocks(sn, en, yn)
    return dict(
        plot=plot,
        plot_kw=plot_kw,
        post=post,
        n_pts=n_pts,
        post_frac_bottom=post_frac_bottom,
        corner_blocks=corner_blocks,
        full_bottom_prog=full_bottom_prog,
        right_prog=right_prog,
        right_blocks=right_blocks,
        shell_cache_key=f"{_CH5_SEQ_SHELL_KEY}:corner_postfrac",
        study=study,
        exam=exam,
        y=y,
        point_order=point_order,
    )


def _build_tutorial_to_prior_landscape_story(
    clip_id, *, compose_id: str = "ch5_01", fixed_end_camera: bool = False,
):
    """After ch5_01: fade tutorial chrome → duo prior landscape (HQ, no 2D legend).

    ``fixed_end_camera=True`` keeps the empty-3D morph and prior landscape at the
    ch5_58 end pose (HQ CT elev/azim) — no W1/W2 camera dance.
    """
    frames = []
    hold = _draft_range(4, 2)
    closing = _ch5_01_closing_state()
    plot_rect = _g("CH4_LIK_PLOT_START_RECT")
    wz, ez, bz = CH5_KNOB_ZERO
    global _CH5_WEIGHT3D_DUO_RECTS, _CH5_SEQ_BARS_DUO_RECTS
    _CH5_WEIGHT3D_DUO_RECTS = None
    _CH5_SEQ_BARS_DUO_RECTS = None
    cfg = ch5_prior_landscape.ch5_prior_landscape_config(hq=True)
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    morph_cam = {}
    if fixed_end_camera:
        morph_cam = dict(
            elev=ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV"))),
            azim=float(_g("CH3_LIK_W12_CT_AZIM")),
        )

    def add(img, n=None):
        frames.extend(_hold(img, hold if n is None else n))

    text_kw = dict(
        bottom_blocks=closing["post_frac_bottom"],
        bottom_prog=closing["full_bottom_prog"],
        corner_blocks=closing["corner_blocks"],
        corner_u=1.0,
        right_prog=closing["right_prog"],
        right_blocks=closing["right_blocks"],
        shell_cache_key=closing["shell_cache_key"],
    )

    # Hold opening beat from ch5_01 end.
    opening = _ch5_compose_bars_frame(
        closing["plot"], compose_id,
        **text_kw,
    )
    add(opening, hold)

    n_fade = _draft_range(CH5_HQ_N_FADE, 6)
    n_pts = int(closing["n_pts"])
    post = closing["post"]
    study, exam, y = closing["study"], closing["exam"], closing["y"]
    empty_mask = np.zeros(len(y), dtype=bool)

    # Fade data, lines, histogram; text stays (keep ch5_01 sequential-bars layout).
    for tv in np.linspace(0.0, 1.0, n_fade, endpoint=True):
        u = float(_g("ch3_knob_smoothstep")(float(tv)))
        n_show = int(round((1.0 - u) * n_pts))
        n_lines = int(round((1.0 - u) * 3))
        n_bars = int(round((1.0 - u) * 3))
        if u >= 1.0 - 1e-6:
            n_show = n_lines = n_bars = 0
        bar_h = post if n_bars > 0 else None
        line_w = post if n_lines > 0 else None
        plot = _cached_frame_sequential_bars(
            n_lines=n_lines, n_bars=n_bars, n_show=n_show,
            bar_heights=bar_h, bar_label_mode="posterior",
            line_weights=line_w,
        )
        frames.append(_ch5_compose_bars_frame(
            plot, compose_id,
            **text_kw,
        ))

    # Histogram → empty 3-D panel (still inside tutorial layout).
    n_hist = _draft_range(CH5_HQ_N_HIST3D, 6)
    for tv in np.linspace(0.0, 1.0, n_hist, endpoint=True):
        u = float(_g("ch3_knob_smoothstep")(float(tv)))
        plot = _frame_ch5_morph_hist_to_weight3d(
            u, ws=wz, we=ez, bb=bz,
            study=study, exam=exam, y=y, mask=empty_mask,
            empty_3d=True, prior_kind="uniform",
            **morph_cam,
        )
        frames.append(_ch5_compose_bars_frame(
            plot, compose_id,
            **text_kw,
        ))

    plot_duo = _frame_ch5_morph_hist_to_weight3d(
        1.0, ws=wz, we=ez, bb=bz,
        study=study, exam=exam, y=y, mask=empty_mask,
        empty_3d=True, prior_kind="uniform",
        **morph_cam,
    )

    # Text rails fade out while the plot stays in the tutorial slot.
    n_text = _draft_range(CH5_HQ_N_TEXT_FADE, 5)
    for tv in np.linspace(1.0, 0.0, n_text, endpoint=True):
        u = float(_g("ch3_knob_smoothstep")(float(tv)))
        frames.append(_ch5_compose_bars_frame(
            plot_duo, compose_id,
            layout_u=1.0,
            panel_u=u,
            title_u=u,
            write_u=u,
            plot_start=plot_rect,
            corner_u=u,
            right_blocks=closing["right_blocks"],
            shell_cache_key=closing["shell_cache_key"],
            bottom_blocks=closing["post_frac_bottom"],
            bottom_prog=closing["full_bottom_prog"],
            corner_blocks=closing["corner_blocks"],
            right_prog=closing["right_prog"],
        ))

    # Plot expands to full canvas as panel rails shrink away (text already gone).
    n_layout = _draft_range(CH5_HQ_N_LAYOUT, 6)
    empty_right_prog = _ch5_seq_right_progress(n_lines=0, n_prior_vals=0)
    for tv in np.linspace(1.0, 0.0, n_layout, endpoint=True):
        u = float(_g("ch3_knob_smoothstep")(float(tv)))
        frames.append(_ch5_compose_bars_frame(
            plot_duo, compose_id,
            layout_u=u,
            panel_u=0.0,
            title_u=0.0,
            write_u=0.0,
            plot_start=plot_rect,
            right_prog=empty_right_prog,
        ))

    add(_ch5_compose_plot_only(plot_duo), hold)

    # Prior landscape (prior_bowl logic) at HQ; uniform prior; knobs zero on rotate + end.
    raw = ch5_prior_landscape.ch5_build_prior_w12_landscape_frames(
        "uniform",
        config=cfg,
        frame_kwargs=fk,
        end_knobs_zero=True,
        knobs_zero_on_rotate=True,
        fixed_end_camera=bool(fixed_end_camera),
    )
    for fr in raw:
        frames.append(_ch5_finish_duo_export(fr, clip_id))
    frames.extend(_hold(frames[-1], hold))
    return frames


def _build_sequential_posterior_landscape_story(clip_id):
    """From HQ prior end: reveal D1 points; posterior landscape + MAP annotation."""
    frames = []
    hold = _draft_range(4, 2)
    pack = _PACKS["D1"]
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    point_order = _ch5_d1_sequential_point_order()
    cfg = ch5_prior_landscape.ch5_prior_landscape_config(hq=True)
    fk = dict(CH5_HQ_LAND_FRAME_KW)

    raw = ch5_prior_landscape.ch5_build_sequential_posterior_w12_frames(
        study, exam, y, point_order,
        prior_kind="uniform",
        config=cfg,
        frame_kwargs=fk,
        annotate_final=True,
        n_seq_hold=int(CH5_HQ_N_SEQ_HOLD),
        n_annot_hold=int(_draft_range(CH5_HQ_N_ANNOT, 4)),
        n_orbit=int(CH5_HQ_GRID_N_ORBIT),
    )
    for fr in raw:
        frames.append(_ch5_finish_duo_export(fr, clip_id))
    frames.extend(_hold(frames[-1], hold))
    return frames


def _ch5_landscape_grid_datasets() -> dict[str, dict]:
    _ensure_packs()
    return {
        k: {
            "study": _PACKS[k]["study"],
            "exam": _PACKS[k]["exam"],
            "y": _PACKS[k]["y"],
            "order": list(_PACKS[k]["order"]),
        }
        for k in CH5_DATASET_KEYS
    }


def _build_uniform_prior_to_sequential_story(clip_id):
    """From ch5_44 rescale handoff: prior landscape choreography + sequential posterior (D1)."""
    frames = []
    hold = _draft_range(4, 2)
    pack = _PACKS["D1"]
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    point_order = _ch5_d1_sequential_point_order()
    cfg = ch5_prior_landscape.ch5_prior_landscape_config(hq=True)
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    wz, ez, bz = CH5_KNOB_ZERO
    empty_mask = np.zeros(len(y), dtype=bool)

    plot_duo = _frame_ch5_morph_hist_to_weight3d(
        1.0, ws=wz, we=ez, bb=bz,
        study=study, exam=exam, y=y, mask=empty_mask,
        empty_3d=True, prior_kind="uniform",
    )
    frames.extend(_hold(_ch5_finish_duo_export(plot_duo, clip_id), hold))

    for fr in ch5_prior_landscape.ch5_build_prior_w12_landscape_frames(
        "uniform", config=cfg, frame_kwargs=fk, end_knobs_zero=True,
    ):
        frames.append(_ch5_finish_duo_export(fr, clip_id))

    for fr in ch5_prior_landscape.ch5_build_sequential_posterior_w12_frames(
        study, exam, y, point_order,
        prior_kind="uniform",
        config=cfg,
        frame_kwargs=fk,
        annotate_final=True,
        n_seq_hold=int(CH5_HQ_N_SEQ_HOLD),
        n_annot_hold=int(_draft_range(CH5_HQ_N_ANNOT, 4)),
    ):
        frames.append(_ch5_finish_duo_export(fr, clip_id))
    frames.extend(_hold(frames[-1], hold))
    return frames


def _build_uniform_landscape_grid_story(clip_id):
    """2×2 HQ grid: cell reveal → synchronized prior build → sequential posterior."""
    hold = _draft_range(4, 2)
    cfg = ch5_prior_landscape.ch5_grid_landscape_config()
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    datasets = _ch5_landscape_grid_datasets()

    def _stream():
        last = None
        for fr in ch5_prior_landscape.ch5_iter_uniform_landscape_grid_frames(
            datasets,
            config=cfg,
            frame_kwargs=fk,
            n_cell_reveal_hold=int(_draft_range(8, 3)),
            n_seq_hold=int(CH5_HQ_N_SEQ_HOLD),
            n_annot_hold=int(_draft_range(CH5_HQ_N_ANNOT, 4)),
            end_knobs_zero=True,
            skip_prior_build=True,
            opening_d1_zoom=True,
        ):
            last = _ch5_finish_duo_export(fr, clip_id)
            yield last
        if last is not None:
            for _ in range(hold):
                yield last

    return _stream()


def _build_uniform_landscape_grid_zoom_story(clip_id):
    """From ch5_47 end: grey inactive cells, zoom each dataset, 360° orbit, zoom back out."""
    frames = []
    hold = _draft_range(4, 2)
    cfg = ch5_prior_landscape.ch5_grid_landscape_config()
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    raw = ch5_prior_landscape.ch5_build_uniform_landscape_grid_zoom_frames(
        _ch5_landscape_grid_datasets(),
        config=cfg,
        frame_kwargs=fk,
    )
    for fr in raw:
        frames.append(_ch5_finish_duo_export(fr, clip_id))
    frames.extend(_hold(frames[-1], hold))
    return frames


def _build_uniform_landscape_grid_focus_story(clip_id):
    """From ch5_47 end: emphasize each dataset by greying inactive quadrants."""
    frames = []
    hold = _draft_range(4, 2)
    cfg = ch5_prior_landscape.ch5_grid_landscape_config()
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    raw = ch5_prior_landscape.ch5_build_uniform_landscape_grid_focus_frames(
        _ch5_landscape_grid_datasets(),
        config=cfg,
        frame_kwargs=fk,
    )
    for fr in raw:
        frames.append(_ch5_finish_duo_export(fr, clip_id))
    frames.extend(_hold(frames[-1], hold))
    return frames


def _build_uniform_landscape_grid_d4_d2_zoom_story(clip_id):
    """From ch5_48 grid: D4 focus → D2+D4 → zoom/orbit D4."""
    frames = []
    hold = _draft_range(4, 2)
    cfg = ch5_prior_landscape.ch5_grid_landscape_config()
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    raw = ch5_prior_landscape.ch5_build_uniform_landscape_grid_d4_d2_zoom_frames(
        _ch5_landscape_grid_datasets(),
        config=cfg,
        frame_kwargs=fk,
    )
    for fr in raw:
        frames.append(_ch5_finish_duo_export(fr, clip_id))
    frames.extend(_hold(frames[-1], hold))
    return frames


def _build_posterior_map_perturb_story(clip_id):
    """MAP line + shadow perturbations (w_EL up, w_ST down) per dataset."""
    frames = []
    hold = _draft_range(4, 2)
    cfg = ch5_prior_landscape.ch5_grid_landscape_config()
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    raw = ch5_prior_landscape.ch5_build_posterior_map_perturb_frames(
        _ch5_landscape_grid_datasets(),
        config=cfg,
        frame_kwargs=fk,
    )
    for fr in raw:
        frames.append(_ch5_finish_duo_export(fr, clip_id))
    frames.extend(_hold(frames[-1], hold))
    return frames


def _build_d4_origin_map_tutorial_story(clip_id):
    """D4 landscape: w_EL/w_ST origin guides → colormap → coupled MAP moves."""
    frames = []
    hold = _draft_range(4, 2)
    cfg = ch5_prior_landscape.ch5_grid_landscape_config()
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    raw = ch5_prior_landscape.ch5_build_d4_origin_map_tutorial_frames(
        _ch5_landscape_grid_datasets(),
        config=cfg,
        frame_kwargs=fk,
    )
    for fr in raw:
        frames.append(_ch5_finish_duo_export(fr, clip_id))
    frames.extend(_hold(frames[-1], hold))
    return frames


def _build_grid_2d_zoom_shadow_orbit_story(clip_id, *, camera_pan: bool = False):
    """2×2 grid: zoom 2D to ±7, D4 swing + shared orbit; optional top-view cam pan."""
    frames = []
    hold = _draft_range(4, 2)
    cfg = ch5_prior_landscape.ch5_grid_landscape_config()
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    raw = ch5_prior_landscape.ch5_build_grid_2d_zoom_shadow_orbit_frames(
        _ch5_landscape_grid_datasets(),
        config=cfg,
        frame_kwargs=fk,
        camera_pan=bool(camera_pan),
    )
    for fr in raw:
        frames.append(_ch5_finish_duo_export(fr, clip_id))
    frames.extend(_hold(frames[-1], hold))
    return frames


def _build_grid_map_labeled_rotate90_story(clip_id):
    """2×2 labeled belief grid, then all plots rotate 90° counter-clockwise."""
    frames = []
    hold = _draft_range(4, 2)
    cfg = ch5_prior_landscape.ch5_grid_landscape_config()
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    raw = ch5_prior_landscape.ch5_build_grid_map_labeled_rotate90_frames(
        _ch5_landscape_grid_datasets(),
        config=cfg,
        frame_kwargs=fk,
    )
    for fr in raw:
        frames.append(_ch5_finish_duo_export(fr, clip_id))
    frames.extend(_hold(frames[-1], hold))
    return frames


def _build_d1_loglik_overlay_story(clip_id):
    """Zoomed D1: clear belief → reveal likelihood → soft belief on top."""
    frames = []
    hold = _draft_range(4, 2)
    cfg = ch5_prior_landscape.ch5_grid_landscape_config()
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    raw = ch5_prior_landscape.ch5_build_d1_loglik_overlay_frames(
        _ch5_landscape_grid_datasets(),
        config=cfg,
        frame_kwargs=fk,
    )
    for fr in raw:
        frames.append(_ch5_finish_duo_export(fr, clip_id))
    frames.extend(_hold(frames[-1], hold))
    return frames


def _build_belief_stem_surface_story(clip_id):
    """End-camera prior: mesh histograms rise → tighten to stems → surface wipe."""
    frames = []
    hold = _draft_range(4, 2)
    cfg = ch5_prior_landscape.ch5_prior_landscape_config(hq=True)
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    raw = ch5_prior_landscape.ch5_build_belief_stem_surface_frames(
        config=cfg,
        frame_kwargs=fk,
        prior_kind="uniform",
    )
    for fr in raw:
        frames.append(_ch5_finish_duo_export(fr, clip_id))
    frames.extend(_hold(frames[-1], hold))
    return frames


def _finish(plot_img, clip_id: str, *, right_blocks=None, show_legend=False, bottom_blocks=None):
    if not _use_text(clip_id):
        return plot_img
    blocks = [] if right_blocks is None else list(right_blocks)
    if show_legend:
        blocks = blocks + _density_legend_blocks()
    return _g("compose_tutorial")(
        plot_img,
        right_blocks=blocks,
        bottom_blocks=bottom_blocks if bottom_blocks is not None else _formula_blocks(),
        corner_blocks=_g("ch4_cached_notation_corner_blocks")(),
        right_title="",
        bottom_title="",
        corner_title="",
        right_title_single_line=True,
        title_write_progress=0.0,
        write_progress=1.0,
        theme="classic_light",
    )


def _frame_fullscreen_text(msg: str):
    fig, ax = plt.subplots(figsize=CH6_FIGSIZE)
    ax.set_axis_off()
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=22, color="#222")
    return ch5_fig_to_image(fig)


def _frame_ch5_dataset(key, *, show_line=False, right_text=None):
    pack = _PACKS[key]
    fig, ax_data, ax_right, axes_k = _g("ch6_figure_duo")()
    _draw_roster_ax(ax_data, pack, show_line=show_line, title=pack["meta"]["title"])
    ws, we, bb = pack["display_w"]
    knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
    _g("ch3_draw_knob_row")(
        fig, axes_k, ws, we, bb, "st", knob_rgbs, canvas_sides,
        rot_strip_deg=0.0, strip_scale=1.0,
        knob_rots=_g("ch3_k1_knob_rots_at")(ws, we, bb), knob_scales=[1.0, 1.0, 1.0], ax_data=ax_data,
    )
    _g("ch6_duo_right_panel_extent")(fig, ax_data, ax_right, axes_k)
    if right_text:
        ax_right.set_axis_off()
        ax_right.text(0.06, 0.92, right_text, transform=ax_right.transAxes, va="top", ha="left", fontsize=14, color="#222")
    else:
        ax_right.set_axis_off()
    return _fig_to_plot(fig)


def _frame_duo_posterior(
    key,
    *,
    colored=True,
    mono=False,
    squish_u=0.0,
    credible_mass=None,
    n_show=None,
    prior_kind="gaussian",
):
    pack = _PACKS[key]
    pk = str(prior_kind).lower()
    fig, ax_data, ax3d, axes_k = _g("ch4_figure_duo_weight3d")()
    if n_show is None:
        n_show = len(pack["order"])
    n_show = int(min(max(0, n_show), len(pack["order"])))
    prior_only = n_show == 0
    credible = credible_mass is not None
    surface = _ch5_w12_surface_for_n(pack, n_show, prior_kind=pk)
    if credible:
        ws, we, bb = _ch5_map_weights(pack, prior_kind=pk)
    else:
        ws, we, bb = surface["ws"], surface["we"], surface["bb"]
    _draw_duo_data_panel(
        ax_data, pack, n_show=n_show, ws=ws, we=we, bb=bb,
        show_threshold=credible,
    )
    _draw_duo_knobs(fig, ax_data, axes_k, ws, we, bb)
    _draw_posterior_surface(
        ax3d, surface,
        colored=colored and not prior_only,
        mono=mono and not prior_only,
        squish_u=squish_u, credible_mass=credible_mass,
        prior_red=prior_only,
        fig=fig, show_map_marker=credible,
        map_ws=ws if credible else None,
        map_we=we if credible else None,
        prior_kind=pk,
    )
    return _fig_to_plot(fig)


_CH5_QUAD_SLOTS = {"D1": (0, 0), "D2": (0, 1), "D3": (1, 0), "D4": (1, 1)}


def _frame_duo_grid_2x2(
    active_keys,
    *,
    credible=False,
    mass=None,
    squish_u=0.0,
    colored=True,
    mono=False,
    prior_kind="gaussian",
):
    """2×2 grid: paste four full single-view duo frames (same layout/DPI as solo)."""
    cells = [[None, None], [None, None]]
    for key in CH5_DATASET_KEYS:
        if key not in active_keys:
            continue
        i, j = _CH5_QUAD_SLOTS[key]
        pack = _PACKS[key]
        cm = None
        if credible:
            cm = float(mass if mass is not None else pack["meta"]["credible_target"])
        cells[i][j] = _frame_duo_posterior(
            key, colored=colored, mono=mono,
            squish_u=squish_u, credible_mass=cm, prior_kind=prior_kind,
        )
    return ch5_composite_2x2_quadrants(cells)


def _frame_credible_2x2(*, mass=None, active_rows=None):
    """2×2 duo grid with credible region highlighted on each 3D floor."""
    keys = list(CH5_DATASET_KEYS if active_rows is None else active_rows)
    return _frame_duo_grid_2x2(keys, credible=True, mass=mass)


def _frame_grid_roster(active_keys):
    """2×2 roster grid; inactive quadrants greyed."""
    fig, axes = ch5_figure_grid(2, 2)
    slots = [("D1", 0, 0), ("D2", 0, 1), ("D3", 1, 0), ("D4", 1, 1)]
    for key, i, j in slots:
        on = key in active_keys
        _draw_roster_ax(axes[i, j], _PACKS[key], show_line=on, alpha=1.0 if on else 0.25, title=_PACKS[key]["meta"]["title"])
        if not on:
            axes[i, j].patch.set_alpha(0.15)
    return ch5_fig_to_image(fig)


def _frame_grid_duo(active_keys, **kw):
    return _frame_duo_grid_2x2(active_keys, **kw)


def _frame_montage_4x4(active_cols=None, active_rows=None):
    """Legacy name — 2×2 duo grid; ``active_rows`` lists lit datasets."""
    keys = list(CH5_DATASET_KEYS if active_rows is None else active_rows)
    return _frame_duo_grid_2x2(keys)


def _frame_credible_4x4(*, mass=None, active_rows=None):
    return _frame_credible_2x2(mass=mass, active_rows=active_rows)


def _ch5_three_line_log_likelihoods(study, exam, y):
    """Log-likelihood for each of the three fixed hypotheses."""
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    nll_fn = _g("_ch3_nll_sum_on_flat_grid")
    out = []
    for ws, we, bb in CH5_THREE_LINES:
        nll = nll_fn(
            study, exam, y,
            np.array([ws], dtype=np.float64),
            np.array([we], dtype=np.float64),
            np.array([bb], dtype=np.float64),
        )
        out.append(-float(np.asarray(nll, dtype=np.float64).ravel()[0]))
    return np.asarray(out, dtype=np.float64)


def _ch5_three_line_belief(study, exam, y, *, stage="posterior"):
    """Discrete prior × likelihood for the three fixed lines."""
    prior = CH5_THREE_LINE_PRIOR.copy()
    if len(study) == 0:
        if stage == "prior":
            return prior
        if stage == "multiply":
            return prior
        return prior
    log_lik = _ch5_three_line_log_likelihoods(study, exam, y)
    lik = np.exp(log_lik - np.max(log_lik))
    if stage == "prior":
        return prior
    if stage == "multiply":
        return prior * np.exp(log_lik)
    log_post = np.log(prior) + log_lik
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    return post / max(float(post.sum()), 1e-12)


def _ch5_d1_sequential_point_order():
    """Reveal order for ch5_02: (3, 2) pass first, then the rest."""
    pack = _PACKS["D1"]
    n = len(pack["y"])
    start = int(CH5_D1_SEQ_START_IDX)
    rest = [i for i in pack["order"] if int(i) != start]
    return [start] + rest


def _ch5_point_prob(ws, we, bb, st, el, y):
    z = float(ws) * float(st) + float(we) * float(el) + float(bb)
    sig = 1.0 / (1.0 + np.exp(-z))
    return float(sig if int(y) == 1 else 1.0 - sig)


def _ch5_line_equation_plain(ws, we, bb):
    ws, we, bb = float(ws), float(we), float(bb)
    if abs(ws - 1.0) < 1e-9 and abs(we + 1.0) < 1e-9:
        body = "x_ST - x_EL"
    else:
        body = f"{ws:g} x_ST {we:+g} x_EL"
    if abs(bb) > 1e-9:
        body += f" {bb:+g}"
    return f"{body} = 0"


def _ch5_line_equation_mathtext(ws, we, bb):
    ws, we, bb = float(ws), float(we), float(bb)
    parts = []
    if abs(ws - 1.0) < 1e-9 and abs(we + 1.0) < 1e-9:
        parts.append(r"x_{\mathrm{ST}} - x_{\mathrm{EL}}")
    else:
        parts.append(
            rf"{ws:g}\, x_{{\mathrm{{ST}}}} {we:+g}\, x_{{\mathrm{{EL}}}}"
        )
    if abs(bb) > 1e-9:
        parts.append(f"{bb:+g}")
    body = " ".join(parts)
    return rf"${body} = 0$"


def _ch5_point_label(st, el, y):
    outcome = "pass" if int(y) == 1 else "fail"
    return f"({float(st):g}, {float(el):g}), {outcome}"


def _ch5_exact_values_blocks(
    study,
    exam,
    y,
):
    """Right-rail: fixed six blocks (3 line equations + 3 value groups). Reveal via progress."""
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n = int(len(study))
    prior_one_third = float(CH5_THREE_LINE_PRIOR[0])
    if n > 0:
        log_lik = _ch5_three_line_log_likelihoods(study, exam, y)
        lik_prod = np.exp(log_lik)
        pt_label = _ch5_point_label(study[-1], exam[-1], y[-1])
    else:
        lik_prod = np.ones(3, dtype=np.float64)
        pt_label = "point"

    blocks: list[dict] = []
    for gi, meta in enumerate(CH5_SEQ_LINE_DISPLAY):
        slot = int(meta["slot"])
        ws, we, bb = CH5_THREE_LINES[slot]
        eq_plain = _ch5_line_equation_plain(ws, we, bb)
        pdata = float(lik_prod[slot])
        color = str(meta["color"])

        title_block = _g("_ch4_formula_hand_block")(
            eq_plain,
            weight=0.0,
            align="left",
            text_x_frac=CH5_SEQ_RIGHT_TEXT_X_FRAC,
            block_fs=CH5_SEQ_LINE_TITLE_FS,
            line_dy_pt=2.0,
            bold_lhs=True,
            role="weights",
            accent_color=color,
            text_color=color,
            pt_units=True,
            top_pad_pt=0.0,
            bottom_pad_pt=0.0,
        )
        if gi > 0:
            title_block["pre_gap_pt"] = CH5_SEQ_LINE_GROUP_GAP_PT
        blocks.append(title_block)

        if n > 0:
            p_new = _ch5_point_prob(ws, we, bb, study[-1], exam[-1], y[-1])
            val_lines = [
                f"P(line) = {_ch5_fmt_prob5(prior_one_third)}",
                f"P({pt_label}  |  line) = {_ch5_fmt_prob5(p_new)}",
                f"P(Data  |  line) = {_ch5_fmt_prob5(pdata)}",
            ]
        else:
            val_lines = [
                f"P(line) = {_ch5_fmt_prob5(prior_one_third)}",
                f"P({pt_label}  |  line) = {_ch5_fmt_prob5(0.0)}",
                f"P(Data  |  line) = {_ch5_fmt_prob5(1.0)}",
            ]
        extra_dy = {0: CH5_SEQ_LINE_VAL_EXTRA_DY_PT, 1: CH5_SEQ_LINE_VAL_EXTRA_DY_PT}
        val_block = _g("_ch4_formula_hand_block")(
            "\n".join(val_lines),
            weight=0.0,
            align="left",
            text_x_frac=CH5_SEQ_RIGHT_TEXT_X_FRAC,
            block_fs=CH5_SEQ_LINE_VAL_FS,
            line_dy_pt=CH5_SEQ_LINE_VAL_LINE_DY_PT,
            line_extra_dy_pt=extra_dy,
            bold_lhs=False,
            role="weights",
            accent_color=color,
            text_color=color,
            pt_units=True,
            top_pad_pt=0.0,
            bottom_pad_pt=0.0,
            pre_gap_pt=CH5_SEQ_LINE_TITLE_VAL_GAP_PT,
            reserve_hidden_lines=True,
        )
        blocks.append(val_block)
    return blocks


def _ch5_seq_bottom_prog(*, bayes: float = 0.0, lik: float = 0.0, prob: float = 0.0) -> dict[int, float]:
    return {0: float(bayes), 1: float(lik), 2: float(prob)}


def _ch5_seq_right_progress(*, n_lines: int = 0, n_prior_vals: int = 0, full_values: bool = False):
    """Per-block line progress for the fixed six-block exact-values rail."""
    prog: dict[int, dict[int, float]] = {}
    n_lines = int(min(max(0, n_lines), 3))
    n_prior_vals = int(min(max(0, n_prior_vals), 3))
    for gi in range(3):
        title_bi = 2 * gi
        val_bi = 2 * gi + 1
        prog[title_bi] = {0: 1.0 if gi < n_lines else 0.0}
        if full_values:
            prog[val_bi] = {0: 1.0, 1: 1.0, 2: 1.0}
        elif gi < n_prior_vals:
            prog[val_bi] = {0: 1.0, 1: 0.0, 2: 0.0}
        else:
            prog[val_bi] = {0: 0.0, 1: 0.0, 2: 0.0}
    return prog


def _ch5_seq_bayes_col_block(*, text="", mathtext_lines=None, mathtext_fs=20.0, **kw):
    drop = float(_g("CH4_FORMULA_BODY_DROP_PT")) + CH5_SEQ_BOTTOM_BAYES_EXTRA_DROP_PT + 20.0
    block = _g("_ch4_formula_hand_block")(
        str(text),
        formula_slot="bayes_col",
        weight=0.30,
        align="left",
        text_x_frac=CH5_SEQ_BAYES_TEXT_X_FRAC,
        block_fs=20.0,
        bold_lhs=False,
        text_y_inset_pt=drop,
        line_dy_pt=5.0,
        pt_units=True,
        role="formula",
    )
    block.update(kw)
    if mathtext_lines is not None:
        block["mathtext_lines"] = [str(ln) for ln in mathtext_lines]
        block["mathtext_fs"] = float(mathtext_fs)
    return block


def _ch5_seq_lik_col(**kw):
    base = dict(
        formula_slot="lik",
        weight=0.26,
        text_x_frac=0.03,
        text_x_shift_pt=CH5_SEQ_LIK_X_SHIFT_PT,
        text_y_inset_pt=CH5_SEQ_LIK_Y_INSET_PT,
        bold_lhs=True,
    )
    base.update(kw)
    return _g("_ch4_formula_hand_block")(CH5_SEQ_DATA_LIK_TEX, **base)


def _ch5_seq_prob_col(**kw):
    shift = float(_g("CH4_FORMULA_CH03_LOG_NLL_X_SHIFT_PT"))
    base = dict(
        formula_slot="log",
        text_x_frac=float(_g("CH4_FORMULA_LOG_TEXT_X_FRAC")),
        text_x_shift_pt=shift + CH5_SEQ_PROB_X_SHIFT_EXTRA_PT,
        text_y_inset_pt=float(_g("CH4_FORMULA_BODY_DROP_PT")),
        bold_lhs=False,
        line_dy_pt=float(_g("CH4_FORMULA_LOG_LINE_DY_PT")),
        cases_row_gap_pt=float(_g("CH4_CASES_ROW_GAP_PT")),
    )
    base.update(kw)
    return _g("_ch4_formula_hand_block")(_g("CH4_PROB_FORMULA_TEX"), **base)


def _ch5_relevant_equations_blocks(
    *,
    bayes_text: str = "New Belief = P(line) × P(Data  |  line)",
    bayes_frac: str | None = None,
    bayes_fs: float = CH5_SEQ_BAYES_FS,
):
    """Bottom-rail: fixed three columns (new belief | likelihood | probability)."""
    bayes_body = str(bayes_frac if bayes_frac is not None else bayes_text)
    bayes_kw = {"text": bayes_body, "block_fs": bayes_fs}
    if bayes_frac is not None:
        drop = (
            float(_g("CH4_FORMULA_BODY_DROP_PT"))
            + CH5_SEQ_BOTTOM_BAYES_EXTRA_DROP_PT
            + 20.0
            - float(CH5_SEQ_BAYES_FRAC_LIFT_PT)
        )
        bayes_kw["text_y_inset_pt"] = drop
    return [
        _ch5_seq_bayes_col_block(**bayes_kw),
        _ch5_seq_lik_col(),
        _ch5_seq_prob_col(),
    ]


def _ch5_fmt_prob5(p):
    return f"{float(p):.5f}"


def _ch5_bottom_formula_block(tex: str | None = None, *, mathtext_lines=None):
    """Centered bottom-rail formula for sequential-bars pedagogy."""
    block = _g("_ch4_formula_hand_block")(
        "" if mathtext_lines else str(tex),
        weight=1.0,
        align="center",
        text_x_frac=0.5,
        block_fs=34.0,
        bold_lhs=False,
        text_y_inset_pt=6.0,
    )
    if mathtext_lines:
        block["mathtext_lines"] = [str(ln) for ln in mathtext_lines]
        block["mathtext_fs"] = 30.0
    return block


def _ch5_finish_bars(
    plot_img,
    clip_id: str,
    *,
    bottom_blocks=None,
    right_blocks=None,
    corner_blocks=None,
    bottom_tex=None,
    bottom_mathtext=None,
    layout_variant: str = "balanced",
    show_corner: bool = False,
    progress_override=None,
    shell_cache_key: str | None = None,
    rails_cache_key: str | None = None,
    layout_u: float = 1.0,
    panel_u: float = 1.0,
    plot_start_rect=None,
    title_write_progress: float = 1.0,
    write_progress: float = 1.0,
):
    if not _use_text(clip_id):
        return plot_img
    if bottom_blocks is None:
        if bottom_mathtext is not None:
            lines = bottom_mathtext if isinstance(bottom_mathtext, (list, tuple)) else [bottom_mathtext]
            bottom_blocks = [_ch5_bottom_formula_block(mathtext_lines=lines)]
        elif bottom_tex is not None:
            bottom_blocks = [_ch5_bottom_formula_block(bottom_tex)]
        else:
            bottom_blocks = []
    right_blocks = [] if right_blocks is None else list(right_blocks)
    if corner_blocks is None:
        corner_blocks = (
            _g("ch4_cached_notation_corner_blocks")()
            if show_corner else []
        )
    compose = _g("ch4_compose_tutorial_frame")
    kw = dict(
        plot_img=plot_img,
        right_blocks=right_blocks,
        bottom_blocks=bottom_blocks,
        corner_blocks=corner_blocks,
        bottom_arrows=None,
        right_title="Exact Values" if right_blocks else "",
        bottom_title="Relevant Equations" if bottom_blocks else "",
        corner_title=_g("CH4_NOTATION_SECTION_TITLE") if corner_blocks else "",
        right_title_single_line=True,
        title_write_progress=float(title_write_progress),
        write_progress=float(write_progress),
        progress_override=progress_override,
        shell_cache_key=shell_cache_key,
        rails_cache_key=rails_cache_key,
        composer=_ch5_seq_composer(layout_variant),
        layout_u=float(layout_u),
        panel_u=float(panel_u),
    )
    if plot_start_rect is not None:
        kw["plot_start_rect"] = plot_start_rect
    return compose(**kw)


def _ch5_n_line_log_likelihoods(study, exam, y, lines):
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    nll_fn = _g("_ch3_nll_sum_on_flat_grid")
    out = []
    for ws, we, bb in lines:
        nll = nll_fn(
            study, exam, y,
            np.array([ws], dtype=np.float64),
            np.array([we], dtype=np.float64),
            np.array([bb], dtype=np.float64),
        )
        out.append(-float(np.asarray(nll, dtype=np.float64).ravel()[0]))
    return np.asarray(out, dtype=np.float64)


def _ch5_five_line_belief(study, exam, y, *, stage="posterior"):
    prior = CH5_FIVE_LINE_PRIOR.copy()
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if len(study) == 0:
        return prior
    log_lik = _ch5_n_line_log_likelihoods(study, exam, y, CH5_FIVE_LINES)
    if stage == "prior":
        return prior
    if stage == "multiply":
        return prior * np.exp(log_lik)
    log_post = np.log(prior) + log_lik
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    return post / max(float(post.sum()), 1e-12)


def _ch5_line_likelihood_weights(study, exam, y, lines):
    """Relative likelihood masses for discrete hypotheses (thickness race)."""
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n = len(lines)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if len(study) == 0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    log_lik = _ch5_n_line_log_likelihoods(study, exam, y, lines)
    lik = np.exp(log_lik - np.max(log_lik))
    s = float(lik.sum())
    return lik / s if s > 1e-12 else np.full(n, 1.0 / n, dtype=np.float64)


def _ch5_parallel_line_family(ws, we, bb, *, n=7, bias_span=1.6):
    """Near-parallel decision boundaries via bias offsets (includes center)."""
    n = max(1, int(n))
    offs = np.linspace(-float(bias_span), float(bias_span), n, endpoint=True)
    return [(float(ws), float(we), float(bb) + float(o)) for o in offs]


def _ch5_lerp_xy_limits(xl0, yl0, xl1, yl1, u):
    """Lerp two (xlim, ylim) pairs; ``u`` in [0, 1]."""
    u = float(_g("ch3_knob_smoothstep")(float(np.clip(u, 0.0, 1.0))))
    xl = (
        float(xl0[0]) + u * (float(xl1[0]) - float(xl0[0])),
        float(xl0[1]) + u * (float(xl1[1]) - float(xl0[1])),
    )
    yl = (
        float(yl0[0]) + u * (float(yl1[0]) - float(yl0[0])),
        float(yl0[1]) + u * (float(yl1[1]) - float(yl0[1])),
    )
    return xl, yl


def _draw_hypothesis_lines_on_ax(
    ax,
    xl,
    yl,
    lines,
    colors,
    *,
    n_lines=None,
    line_weights=None,
    winner_idx=None,
):
    n_lines = len(lines) if n_lines is None else int(min(max(0, n_lines), len(lines)))
    if winner_idx is not None:
        indices = [int(winner_idx)]
    else:
        indices = list(range(n_lines))
    wts = None if line_weights is None else np.asarray(line_weights, dtype=np.float64)
    wmax = max(float(wts.max()), 1e-12) if wts is not None and wts.size else 1.0
    for i in indices:
        ws, we, bb = lines[i]
        bxy = _g("boundary_line_xy")(ws, we, bb, *xl, *yl)
        if bxy is None:
            continue
        bx, by = bxy
        if winner_idx is not None:
            color, linestyle, lw, alpha = "0.25", "--", 2.0, 0.95
        else:
            color = colors[i]
            linestyle = "-"
            wt = 1.0 if wts is None else float(wts[i])
            lw = 1.2 + 3.2 * float(wt / wmax)
            alpha = 0.40 + 0.60 * float(wt / wmax)
        ax.plot(bx, by, c=color, linewidth=lw, linestyle=linestyle, alpha=alpha, zorder=3)


def _draw_three_lines_on_ax(ax, xl, yl, *, n_lines=3, line_weights=None):
    n_lines = int(min(max(0, n_lines), 3))
    wts = np.asarray([1.0 / 3.0] * 3 if line_weights is None else line_weights, dtype=np.float64)
    wmax = max(float(wts.max()), 1e-12)
    for j in range(n_lines):
        i = int(CH5_SEQ_LINE_REVEAL_IDX[j])
        (ws, we, bb), color = CH5_THREE_LINES[i], CH5_THREE_LINE_COLORS[i]
        bxy = _g("boundary_line_xy")(ws, we, bb, *xl, *yl)
        if bxy is not None:
            bx, by = bxy
            wt = float(wts[i])
            lw = 1.4 + 2.6 * float(wt / wmax) if line_weights is not None else 2.2
            alpha = 0.45 + 0.55 * float(wt / wmax) if line_weights is not None else 0.88
            ax.plot(bx, by, c=color, linewidth=lw, alpha=alpha, zorder=3)


def _draw_sequential_bars_on_ax(
    ax,
    *,
    n_bars=0,
    heights=None,
    label_mode=None,
    n_labels=0,
    bar_ymax=None,
):
    """Fixed-slot bar chart: yellow left, blue middle, purple right."""
    n_bars = int(min(max(0, n_bars), 3))
    n_labels = int(min(max(0, n_labels), n_bars))
    if heights is None:
        heights = [1.0 / 3.0] * 3
    heights = np.asarray(heights, dtype=np.float64)
    ymax = float(bar_ymax) if bar_ymax is not None else CH5_SEQ_BAR_YMAX
    bar_heights = [float(heights[CH5_SEQ_BAR_LINE_IDX[i]]) for i in range(n_bars)]
    if label_mode == "multiply":
        ymax = max(ymax, max(bar_heights) * 1.28 + 0.05)
    elif label_mode in ("posterior", "prior"):
        ymax = max(ymax, max(bar_heights[:n_bars], default=0) * 1.15 + 0.06)
    x = np.asarray(CH5_SEQ_BAR_SLOTS, dtype=float)
    for slot in range(n_bars):
        line_i = int(CH5_SEQ_BAR_LINE_IDX[slot])
        h = bar_heights[slot]
        ax.bar(
            x[slot], h,
            width=CH5_SEQ_BAR_WIDTH,
            color=CH5_THREE_LINE_COLORS[line_i],
            align="center",
            zorder=2,
        )
        if label_mode is None or slot >= n_labels:
            continue
        if label_mode == "prior":
            label = r"$P(\mathrm{line}) = \frac{1}{3}$"
        elif label_mode == "multiply":
            label = (
                r"$P(\mathrm{line})\, P(\mathrm{Data}\mid\mathrm{line})$" + "\n"
                + rf"$= {_ch5_fmt_prob5(h)}$"
            )
        elif label_mode == "posterior":
            label = (
                r"$P(\mathrm{line} \mid \mathrm{Data})$" + "\n"
                + rf"$= {_ch5_fmt_prob5(h)}$"
            )
        else:
            label = ""
        if label:
            ax.text(
                x[slot], h + 0.035 * ymax, label,
                ha="center", va="bottom", fontsize=CH5_SEQ_BAR_LABEL_FS, color="#222", zorder=4,
            )
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _frame_sequential_bars(
    *,
    n_lines=0,
    n_bars=0,
    n_prior_labels=0,
    n_show=0,
    bar_heights=None,
    bar_label_mode=None,
    line_weights=None,
    point_order=None,
):
    """2D + knobs | histogram — sequential three-line Bayesian update."""
    pack = _PACKS["D1"]
    if point_order is None:
        point_order = _ch5_d1_sequential_point_order()
    n_show = int(min(max(0, n_show), len(point_order)))
    xl, yl = ch5_plot_limits("D1")
    fig, ax_data, ax_right, axes_k = _g("ch6_figure_duo")()

    mask = np.zeros(len(pack["y"]), dtype=bool)
    for j in point_order[:n_show]:
        mask[int(j)] = True

    def _lines(ax, _xl, _yl):
        _draw_three_lines_on_ax(
            ax, _xl, _yl, n_lines=n_lines, line_weights=line_weights,
        )

    _ch5_draw_data_panel(
        ax_data, pack["study"], pack["exam"], pack["y"],
        xl=xl, yl=yl, mask=mask, after_draw=_lines,
    )
    ws, we, bb = CH5_THREE_LINES[1]
    knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
    _g("ch3_draw_knob_row")(
        fig, axes_k, ws, we, bb, "all", knob_rgbs, canvas_sides,
        rot_strip_deg=0.0, strip_scale=1.0,
        knob_rots=_g("ch3_k1_knob_rots_at")(ws, we, bb), knob_scales=[1.0, 1.0, 1.0],
        ax_data=ax_data,
    )
    _g("_ch3_align_knob_axes_under_data")(fig, ax_data, axes_k)
    _g("ch3_layout_knob_axes_like_bridge_end")(fig, ax_data, axes_k)
    _g("ch6_duo_right_panel_extent")(fig, ax_data, ax_right, axes_k)
    _draw_sequential_bars_on_ax(
        ax_right,
        n_bars=n_bars,
        heights=bar_heights,
        label_mode=bar_label_mode,
        n_labels=(
            n_prior_labels if bar_label_mode == "prior"
            else (n_bars if bar_label_mode in ("posterior", "multiply") else 0)
        ),
        bar_ymax=CH5_SEQ_BAR_YMAX_POST if bar_label_mode == "posterior" else None,
    )
    return _fig_to_plot(fig)


def _ch5_seq_progress_key(prog: dict[int, dict]) -> tuple:
    return tuple(
        (int(bi), tuple(sorted((int(li), float(p)) for li, p in lp.items())))
        for bi, lp in sorted(prog.items())
    )


def _ch5_seq_compose_cache_key(
    *,
    plot_kw,
    right_prog,
    bottom_prog,
    bottom_blocks,
    corner_u,
    shell_key,
    rails_key,
    n_show,
) -> tuple:
    bottom_text = tuple(str(b.get("text", "")) for b in (bottom_blocks or []))
    return (
        _ch5_seq_plot_cache_key(**plot_kw),
        _ch5_seq_progress_key(right_prog),
        None if bottom_prog is None else tuple(sorted((int(k), float(v)) for k, v in bottom_prog.items())),
        bottom_text,
        float(corner_u),
        shell_key,
        rails_key,
        int(n_show),
    )


def _ch5_seq_plot_cache_key(**kw) -> tuple:
    items: list[tuple] = []
    for key in sorted(kw):
        val = kw[key]
        if isinstance(val, np.ndarray):
            val = tuple(float(x) for x in val.ravel())
        items.append((key, val))
    return tuple(items)


def _cached_frame_sequential_bars(**kw):
    key = _ch5_seq_plot_cache_key(**kw)
    cached = _CH5_SEQ_PLOT_CACHE.get(key)
    if cached is None:
        cached = _frame_sequential_bars(**kw)
        _CH5_SEQ_PLOT_CACHE[key] = cached
    return cached


def ch5_preview_seq_layout_frame(
    *,
    layout_variant: str = "balanced",
    bayes_fs: float | None = None,
    show_fraction: bool = False,
    n_show: int = 1,
    bar_label_mode: str = "multiply",
):
    """Single composed frame for ch5_01 layout previews (multiply + full rails)."""
    pack = _PACKS["D1"]
    point_order = _ch5_d1_sequential_point_order()
    n_show = int(min(max(0, n_show), len(point_order)))
    idxs = point_order[:n_show]
    study, exam, y = pack["study"][idxs], pack["exam"][idxs], pack["y"][idxs]
    if bar_label_mode == "posterior":
        heights = _ch5_three_line_belief(study, exam, y, stage="posterior")
        line_weights = heights
    else:
        heights = _ch5_three_line_belief(study, exam, y, stage="multiply")
        line_weights = None
    fs_presets = {"compact": 21.0, "balanced": CH5_SEQ_BAYES_FS, "airy": 25.0}
    fs = float(bayes_fs if bayes_fs is not None else fs_presets.get(layout_variant, 20.0))
    plot = _frame_sequential_bars(
        n_lines=3, n_bars=3, n_show=n_show,
        bar_heights=heights, bar_label_mode=bar_label_mode,
        line_weights=line_weights,
    )
    bottom = _ch5_relevant_equations_blocks(
        bayes_frac=CH5_SEQ_BAYES_FRAC_TEX if show_fraction else None,
        bayes_text="New Belief = P(line) × P(Data  |  line)",
        bayes_fs=fs,
    )
    return _ch5_finish_bars(
        plot, "ch5_01",
        bottom_blocks=bottom,
        right_blocks=_ch5_exact_values_blocks(study, exam, y),
        layout_variant=layout_variant,
    )


def _build_sequential_bars_story(clip_id):
    """Staged reveal: lines → bars/P(line) → Bayes text → likelihood → notation → data."""
    pack = _PACKS["D1"]
    point_order = _ch5_d1_sequential_point_order()
    hold = _draft_range(6, 3)
    hold_short = _draft_range(4, 2)
    frames = []
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    empty = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)
    corner_blocks = _g("ch4_cached_notation_corner_blocks")()
    style = _ch5_seq_composer("balanced").handwrite_style()
    _CH5_SEQ_PLOT_CACHE.clear()
    _CH5_SEQ_COMPOSE_CACHE.clear()
    _ch5_seq_composer("balanced").clear_rails_cache()

    def _subset(n_show):
        idxs = point_order[: int(n_show)]
        return study[idxs], exam[idxs], y[idxs]

    def _right_blocks(n_show):
        sn, en, yn = _subset(n_show) if n_show > 0 else (empty, empty, empty_y)
        return _ch5_exact_values_blocks(sn, en, yn)

    def _emit(img, n):
        frames.extend([img] * int(n))

    def _compose(
        *,
        plot_kw,
        n_lines=0,
        n_prior_vals=0,
        full_values=False,
        bottom_blocks=None,
        bottom_prog=None,
        corner_u=0.0,
        corner=None,
        shell_key=None,
        rails_key=None,
    ):
        plot = _cached_frame_sequential_bars(**plot_kw)
        right_blocks = _right_blocks(int(plot_kw.get("n_show", 0)))
        right_prog = _ch5_seq_right_progress(
            n_lines=n_lines,
            n_prior_vals=n_prior_vals,
            full_values=full_values,
        )
        po = {"right": right_prog}
        if bottom_blocks is not None and bottom_prog is not None:
            po["bottom"] = ch4_bottom_per_block_progress(bottom_blocks, bottom_prog)
        if corner is not None:
            po["corner"] = ch4_blocks_write_from_slot(corner, 0, float(corner_u), style=style)
        cache_key = _ch5_seq_compose_cache_key(
            plot_kw=plot_kw,
            right_prog=right_prog,
            bottom_prog=bottom_prog,
            bottom_blocks=bottom_blocks,
            corner_u=corner_u if corner is not None else 0.0,
            shell_key=shell_key,
            rails_key=rails_key,
            n_show=int(plot_kw.get("n_show", 0)),
        )
        cached = _CH5_SEQ_COMPOSE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        img = _ch5_finish_bars(
            plot,
            clip_id,
            bottom_blocks=bottom_blocks,
            right_blocks=right_blocks,
            corner_blocks=corner if corner is not None else None,
            show_corner=corner is not None,
            progress_override=po,
            shell_cache_key=shell_key,
            rails_cache_key=rails_key,
        )
        _CH5_SEQ_COMPOSE_CACHE[cache_key] = img
        return img

    bayes_rhs_steps = [
        "New Belief = ",
        "New Belief = prior belief",
        "New Belief = prior belief × likelihood",
        "New Belief = P(line) × P(Data | line)",
    ]
    bayes_full = bayes_rhs_steps[-1]
    bayes_only_prog = _ch5_seq_bottom_prog(bayes=1.0)
    lik_prog = _ch5_seq_bottom_prog(bayes=1.0, lik=1.0)
    full_bottom_prog = _ch5_seq_bottom_prog(bayes=1.0, lik=1.0, prob=1.0)

    # Empty panels, knobs visible.
    _emit(_compose(plot_kw=dict(n_lines=0, n_bars=0)), hold_short)

    # Lines + line equations, yellow → blue → purple.
    for k in range(1, 4):
        _emit(_compose(
            plot_kw=dict(n_lines=k, n_bars=0),
            n_lines=k,
            rails_key=_CH5_SEQ_RAILS_KEY,
        ), hold_short)

    # Prior bars + P(line) = 1/3 on each bar, same reveal order.
    for k in range(1, 4):
        _emit(_compose(
            plot_kw=dict(
                n_lines=3, n_bars=k,
                bar_label_mode="prior", n_prior_labels=k,
            ),
            n_lines=3,
            n_prior_vals=k,
            rails_key=_CH5_SEQ_RAILS_KEY,
        ), hold_short)

    prior_bar_kw = dict(n_lines=3, n_bars=3, bar_label_mode="prior", n_prior_labels=3)

    # Bottom rail: New Belief = → prior belief → × likelihood → P(line)×P(Data|line).
    for tex in bayes_rhs_steps:
        blocks = _ch5_relevant_equations_blocks(bayes_text=tex)
        _emit(_compose(
            plot_kw=dict(prior_bar_kw),
            n_lines=3,
            n_prior_vals=3,
            bottom_blocks=blocks,
            bottom_prog=bayes_only_prog,
            shell_key=_CH5_SEQ_SHELL_KEY,
        ), hold)

    # P(Data | line) column.
    lik_bottom = _ch5_relevant_equations_blocks(bayes_text=bayes_full)
    _emit(_compose(
        plot_kw=dict(prior_bar_kw),
        n_lines=3,
        n_prior_vals=3,
        bottom_blocks=lik_bottom,
        bottom_prog=lik_prog,
        shell_key=_CH5_SEQ_SHELL_KEY,
    ), hold)

    # p(yᵢ | xᵢ) column.
    full_bottom = _ch5_relevant_equations_blocks(bayes_text=bayes_full)

    _emit(_compose(
        plot_kw=dict(prior_bar_kw),
        n_lines=3,
        n_prior_vals=3,
        bottom_blocks=full_bottom,
        bottom_prog=full_bottom_prog,
        shell_key=_CH5_SEQ_SHELL_KEY,
    ), hold)

    # Notation corner.
    _emit(_compose(
        plot_kw=dict(prior_bar_kw),
        n_lines=3,
        n_prior_vals=3,
        bottom_blocks=full_bottom,
        bottom_prog=full_bottom_prog,
        corner=corner_blocks,
        corner_u=1.0,
        shell_key=f"{_CH5_SEQ_SHELL_KEY}:corner",
    ), hold)

    # First data point — multiply bars + full exact values on right.
    s1, e1, y1 = _subset(1)
    unnorm1 = _ch5_three_line_belief(s1, e1, y1, stage="multiply")
    post1 = _ch5_three_line_belief(s1, e1, y1, stage="posterior")
    _emit(_compose(
        plot_kw=dict(
            n_lines=3, n_bars=3, n_show=1,
            bar_heights=unnorm1, bar_label_mode="multiply",
        ),
        n_lines=3,
        full_values=True,
        bottom_blocks=full_bottom,
        bottom_prog=full_bottom_prog,
        corner=corner_blocks,
        corner_u=1.0,
        shell_key=f"{_CH5_SEQ_SHELL_KEY}:corner",
    ), hold)

    # New Belief with fraction RHS (still unnormalized bars).
    new_frac_bottom = _ch5_relevant_equations_blocks(bayes_frac=CH5_SEQ_BAYES_NEW_FRAC_TEX)
    _emit(_compose(
        plot_kw=dict(
            n_lines=3, n_bars=3, n_show=1,
            bar_heights=unnorm1, bar_label_mode="multiply",
        ),
        n_lines=3,
        full_values=True,
        bottom_blocks=new_frac_bottom,
        bottom_prog=full_bottom_prog,
        corner=corner_blocks,
        corner_u=1.0,
        shell_key=f"{_CH5_SEQ_SHELL_KEY}:corner_newfrac",
    ), hold)

    # P(line | Data) fraction + normalized posterior bars.
    post_frac_bottom = _ch5_relevant_equations_blocks(bayes_frac=CH5_SEQ_BAYES_FRAC_TEX)
    _emit(_compose(
        plot_kw=dict(
            n_lines=3, n_bars=3, n_show=1,
            bar_heights=post1, bar_label_mode="posterior",
            line_weights=post1,
        ),
        n_lines=3,
        full_values=True,
        bottom_blocks=post_frac_bottom,
        bottom_prog=full_bottom_prog,
        corner=corner_blocks,
        corner_u=1.0,
        shell_key=f"{_CH5_SEQ_SHELL_KEY}:corner_postfrac",
    ), hold)

    # Remaining data points — update bars and exact values.
    for n in range(2, len(point_order) + 1):
        sn, en, yn = _subset(n)
        post = _ch5_three_line_belief(sn, en, yn, stage="posterior")
        _emit(_compose(
            plot_kw=dict(
                n_lines=3, n_bars=3, n_show=n,
                bar_heights=post, bar_label_mode="posterior",
                line_weights=post,
            ),
            n_lines=3,
            full_values=True,
            bottom_blocks=post_frac_bottom,
            bottom_prog=full_bottom_prog,
            corner=corner_blocks,
            corner_u=1.0,
            shell_key=f"{_CH5_SEQ_SHELL_KEY}:corner_postfrac",
        ), hold_short)

    frames.extend(_hold(frames[-1]))
    return frames


def _frame_three_lines_panel(*, n_show, bar_vals=None, line_weights=None):
    """D1 panel: three colored boundaries + sequential point reveal; optional bar chart."""
    pack = _PACKS["D1"]
    order = pack["order"]
    n_show = int(min(max(0, n_show), len(order)))
    xl, yl = ch5_plot_limits("D1")
    fig, ax_data, ax_right, axes_k = _g("ch6_figure_duo")()
    mask = np.zeros(len(pack["y"]), dtype=bool)
    for j in order[:n_show]:
        mask[int(j)] = True

    def _lines(ax, _xl, _yl):
        _draw_three_lines_on_ax(ax, _xl, _yl, n_lines=3, line_weights=line_weights)

    _ch5_draw_data_panel(
        ax_data, pack["study"], pack["exam"], pack["y"],
        xl=xl, yl=yl, mask=mask, after_draw=_lines,
    )
    for ax in axes_k:
        ax.set_axis_off()
    _g("ch6_duo_right_panel_extent")(fig, ax_data, ax_right, axes_k)
    ax_right.set_axis_off()
    if bar_vals is not None:
        vals = np.asarray(bar_vals, dtype=np.float64)
        x = np.arange(3)
        ax_right.bar(x, vals, color=list(CH5_THREE_LINE_COLORS))
        ax_right.set_xticks(x, CH5_THREE_LINE_LABELS)
        ax_right.set_ylim(0, max(float(vals.max()) * 1.15, 0.4))
    return _fig_to_plot(fig)


def _frame_discrete_bars(stage, *, n_pts=0):
    """Three-line discrete belief: prior / multiply / normalize bars on Ch5 right."""
    pack = _PACKS["D1"]
    order = pack["order"]
    if stage == "prior":
        n_show = 0
        vals = _ch5_three_line_belief(pack["study"][:0], pack["exam"][:0], pack["y"][:0], stage="prior")
    elif stage in ("multiply", "normalize"):
        n_show = len(order)
        study, exam, y, _ = _ch5_subset_pack(pack, n_show)
        vals = _ch5_three_line_belief(study, exam, y, stage=stage)
    else:
        n_show = int(min(max(0, n_pts), len(order)))
        study, exam, y, _ = _ch5_subset_pack(pack, n_show)
        vals = _ch5_three_line_belief(study, exam, y, stage="posterior")
    wts = vals / max(float(np.asarray(vals).sum()), 1e-12)
    return _frame_three_lines_panel(n_show=n_show, bar_vals=vals, line_weights=wts)


def _build_sequential_three_lines(clip_id, *, show_bars=False):
    """Deprecated — use ``_build_sequential_bars_story``."""
    return _build_sequential_bars_story(clip_id)


def _frame_prior_1d():
    pack = _PACKS["D1"]
    fig, ax_data, ax_right, axes_k = _g("ch6_figure_duo")()
    w = np.linspace(CH5_VIEW_BOUNDS[0], CH5_VIEW_BOUNDS[1], 80)
    lp = ch5_log_prior(w, np.zeros_like(w), np.zeros_like(w))
    d = ch5_density_from_log(lp)
    ax_data.plot(w, d, color="#446688", lw=2)
    ax_data.set_xlabel(r"$w_{\mathrm{ST}}$")
    ax_data.set_ylabel(r"$p(w_{\mathrm{ST}})$")
    for ax in axes_k:
        ax.set_axis_off()
    _g("ch6_duo_right_panel_extent")(fig, ax_data, ax_right, axes_k)
    ax_right.set_axis_off()
    return _fig_to_plot(fig)


def _frame_nll_bridge():
    pack = _PACKS["D1"]
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    W1, W2 = pack["W1"], pack["W2"]
    B = np.full_like(W1, CH5_W12_B_FIXED)
    nll_fn = _g("_ch3_nll_sum_on_flat_grid")
    log_post = ch5_log_posterior_grid(study, exam, y, W1.ravel(), W2.ravel(), B.ravel(), nll_fn=nll_fn)
    nll = ch5_log_likelihood_grid(study, exam, y, W1.ravel(), W2.ravel(), B.ravel(), nll_fn=nll_fn)
    nll = (-nll).reshape(W1.shape)
    post_d = ch5_density_from_log(log_post.reshape(W1.shape))
    nll_d = ch5_density_from_log(nll)
    fig = plt.figure(figsize=CH6_FIGSIZE)
    for Z in (nll_d, post_d):
        ax3d = fig.add_subplot(1, 2, len(fig.axes) + 1, projection="3d")
        _style_ax3d_mini(ax3d)
        _draw_density_surface(ax3d, W1, W2, Z, colored=True)
    fig.subplots_adjust(left=0.02, right=0.99, wspace=0.08)
    return ch5_fig_to_image(fig)


def _frame_ct_single(
    key,
    *,
    sweep_axis="el",
    plane_val=0.0,
    pivot_from=None,
    pivot_to=None,
    pivot_u=0.0,
    n_show=None,
    prior_kind=None,
    ct_gn=None,
    show_map=True,
    show_threshold=True,
    hq_elev=False,
):
    pack = _PACKS[key]
    bounds = CH5_VIEW_BOUNDS
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = bounds
    xl, yl = ch5_plot_limits(key)
    pk = "gaussian" if prior_kind is None else str(prior_kind)

    if n_show is not None or prior_kind is not None:
        study, exam, y, n_vis = _ch5_subset_pack(pack, n_show if n_show is not None else len(pack["order"]))
    else:
        study, exam, y = pack["study"], pack["exam"], pack["y"]
        n_vis = len(y)

    w1, w2, b, log_post, log_prior = _ch5_ct_mesh_and_log_post(
        study, exam, y, bounds,
        sweep_axis=sweep_axis,
        plane_val=plane_val,
        pivot_from=pivot_from,
        pivot_to=pivot_to,
        pivot_u=pivot_u,
        prior_kind=pk,
        gn=ct_gn,
    )
    prior_only = n_vis == 0
    # Outside the plausible region (posterior ≡ 0 / −∞ on the whole slice): no MAP chrome.
    plausible = (not prior_only) and _ch5_ct_slice_has_plausible_mass(log_post)
    if plausible:
        map_ws, map_we, map_bb = _ch5_ct_slice_map_from_mesh(w1, w2, b, log_post)
        show_map_eff = bool(show_map)
        show_thr_eff = bool(show_threshold)
    else:
        map_ws, map_we, map_bb = float(CH5_KNOB_ZERO[0]), float(CH5_KNOB_ZERO[1]), float(CH5_KNOB_ZERO[2])
        show_map_eff = False
        show_thr_eff = False

    fig, ax_data, ax3d, axes_k = _g("ch4_figure_duo_weight3d")()
    if n_show is None and prior_kind is None:
        _ch5_draw_data_panel(
            ax_data, study, exam, y, xl=xl, yl=yl,
            ws=map_ws if show_thr_eff else None,
            we=map_we if show_thr_eff else None,
            bb=map_bb if show_thr_eff else None,
            show_colormap=False, show_threshold=show_thr_eff,
            legend_emphasis="all",
        )
    else:
        mask = np.zeros(len(pack["y"]), dtype=bool)
        for j in pack["order"][:n_vis]:
            mask[int(j)] = True
        _ch5_draw_data_panel(
            ax_data, pack["study"], pack["exam"], pack["y"],
            xl=xl, yl=yl, mask=mask,
            ws=map_ws if show_thr_eff else None,
            we=map_we if show_thr_eff else None,
            bb=map_bb if show_thr_eff else None,
            show_colormap=False, show_threshold=show_thr_eff,
            legend_emphasis="all",
        )
    knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
    _g("ch3_draw_knob_row")(
        fig, axes_k, map_ws, map_we, map_bb, "all", knob_rgbs, canvas_sides,
        rot_strip_deg=0.0, strip_scale=1.0,
        knob_rots=_g("ch3_k1_knob_rots_at")(map_ws, map_we, map_bb),
        knob_scales=[1.0, 1.0, 1.0], ax_data=ax_data,
    )
    _g("_ch3_lik_style_ax3d")(ax3d, dlo1, dhi1, dlo2, dhi2, dlob, dhib)
    _ch5_ct_view_init(ax3d, hq_elev=bool(hq_elev))
    _ch5_ct_draw_slice(ax3d, w1, w2, b, log_post, log_prior=log_prior, prior_red=prior_only, prior_kind=pk)
    if show_map_eff:
        _ch5_ct_draw_map_on_slice(fig, ax3d, map_ws, map_we, map_bb)
    return _fig_to_plot(fig)


def _frame_ct_grid_2x2(
    active_keys,
    *,
    sweep_axis="el",
    plane_val=0.0,
    pivot_from=None,
    pivot_to=None,
    pivot_u=0.0,
    n_show=None,
    prior_kind=None,
    ct_gn=None,
    show_map=True,
    show_threshold=True,
    hq_elev=False,
):
    """2×2 grid of CT scans — same slice pose for each dataset."""
    cells = [[None, None], [None, None]]
    for key in CH5_DATASET_KEYS:
        if key not in active_keys:
            continue
        i, j = _CH5_QUAD_SLOTS[key]
        cells[i][j] = _frame_ct_single(
            key,
            sweep_axis=sweep_axis,
            plane_val=plane_val,
            pivot_from=pivot_from,
            pivot_to=pivot_to,
            pivot_u=pivot_u,
            n_show=n_show,
            prior_kind=prior_kind,
            ct_gn=ct_gn,
            show_map=show_map,
            show_threshold=show_threshold,
            hq_elev=hq_elev,
        )
    return ch5_composite_2x2_quadrants(cells)


def _build_hq_uniform_ct_grid_frames(
    active_keys=None,
    *,
    prior_kind: str = "uniform",
    start_b: float | None = None,
) -> list:
    """
    HQ 2×2 CT scan starting on the b=start_b plane (default 0).

    Motion: hold at b → scan down b to lo → continuous pivot b→st → st sweep →
    pivot st→el → el sweep → pivot el→b → full b sweep. Every plane transition
    is a continuous pivot (no hard jumps).
    """
    keys = list(CH5_DATASET_KEYS if active_keys is None else active_keys)
    pk = str(prior_kind)
    frames: list = []
    bounds = CH5_VIEW_BOUNDS
    b0 = float(CH5_W12_B_FIXED if start_b is None else start_b)
    ct_axes = ("st", "el", "b")
    pivot_map = {"b": "st", "st": "el", "el": "b"}
    n_hold = int(CH5_HQ_CT_N_HOLD)
    gn = int(CH5_HQ_CT_GRID)
    n_pivot = int(CH5_HQ_CT_N_PIVOT)
    n_sweep = int(CH5_HQ_CT_N_SWEEP)
    ct_kw = dict(
        prior_kind=pk, ct_gn=gn, show_map=True, show_threshold=True, hq_elev=True,
    )
    blo, _bhi = _g("_ch4_ct_axis_limits")("b", bounds)
    smooth = _g("ch3_knob_smoothstep")

    # Opening: squished surface held on the bias axis at b=0 (or start_b).
    for _ in range(n_hold):
        img = _frame_ct_grid_2x2(keys, sweep_axis="b", plane_val=b0, **ct_kw)
        frames.append(img)

    # Scan down the b axis from start_b → lo (continuous approach to the pivot edge).
    if abs(float(b0) - float(blo)) > 1e-9:
        for tv in np.linspace(0.0, 1.0, n_sweep, endpoint=True):
            u = float(smooth(float(tv)))
            val = float(b0) + u * (float(blo) - float(b0))
            img = _frame_ct_grid_2x2(keys, sweep_axis="b", plane_val=val, **ct_kw)
            frames.append(img)

    # Across: b→st pivot, then st / el / b with existing continuous pivots.
    scan_axes = ("b",) + ct_axes  # logical prev for first pivot is b
    for i, axis in enumerate(ct_axes):
        prev = scan_axes[i]  # b before st, st before el, el before b
        nxt = pivot_map.get(prev)
        if nxt == axis:
            for tv in np.linspace(0.0, 1.0, n_pivot, endpoint=True):
                img = _frame_ct_grid_2x2(
                    keys,
                    pivot_from=prev,
                    pivot_to=axis,
                    pivot_u=float(tv),
                    **ct_kw,
                )
                frames.append(img)
        lo, hi = _g("_ch4_ct_axis_limits")(axis, bounds)
        for tv in np.linspace(0.0, 1.0, n_sweep, endpoint=True):
            u = float(smooth(float(tv)))
            val = lo + u * (hi - lo)
            img = _frame_ct_grid_2x2(keys, sweep_axis=axis, plane_val=val, **ct_kw)
            frames.append(img)
    return frames


def _build_uniform_landscape_squish_ct_story(clip_id):
    """From ch5_47 end: squish belief → CT at b=0 → continuous 2×2 HQ CT scan."""
    frames = []
    hold = _draft_range(4, 2)
    cfg = ch5_prior_landscape.ch5_grid_landscape_config()
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    datasets = _ch5_landscape_grid_datasets()

    squish_raw = ch5_prior_landscape.ch5_build_uniform_landscape_squish_frames(
        datasets, config=cfg, frame_kwargs=fk,
    )
    # Brief hold on the unsquished 47+ surfaces before compressing.
    if squish_raw:
        frames.extend(_hold(_ch5_finish_duo_export(squish_raw[0], clip_id), hold))
    for fr in squish_raw:
        frames.append(_ch5_finish_duo_export(fr, clip_id))

    ct_hold = _frame_ct_grid_2x2(
        CH5_DATASET_KEYS,
        sweep_axis="b",
        plane_val=float(CH5_W12_B_FIXED),
        prior_kind="uniform",
        ct_gn=int(CH5_HQ_CT_GRID),
        show_map=True,
        show_threshold=True,
        hq_elev=True,
    )
    n_morph = int(_draft_range(CH5_HQ_GRID_N_CAM_MORPH, 4))
    squish_end = squish_raw[-1]
    for tv in np.linspace(0.0, 1.0, n_morph, endpoint=True):
        u = float(_g("ch3_knob_smoothstep")(float(tv)))
        morph = ch5_crossfade_images(squish_end, ct_hold, u)
        frames.append(_ch5_finish_duo_export(morph, clip_id))

    for fr in _build_hq_uniform_ct_grid_frames(CH5_DATASET_KEYS, prior_kind="uniform"):
        frames.append(_ch5_finish_duo_export(fr, clip_id))
    frames.extend(_hold(frames[-1], hold))
    return frames


def _ch5_posterior_3d_voxel_data(study, exam, y, *, prior_kind="gaussian", gn=None, mass=None):
    """3D HPD credible voxels: highest-density cells covering ``mass`` of belief."""
    return ch5_posterior_3d_pack(
        study, exam, y,
        prior_kind=prior_kind, gn=gn, mass=mass,
        nll_fn=_g("_ch3_nll_sum_on_flat_grid"),
    )


def _draw_squished_floor_landscape(ax3d, surface, *, z_floor=None, prior_kind="gaussian"):
    """Squished belief landscape on the (w_ST, w_EL) floor in parameter space."""
    W1, W2, density = surface["W1"], surface["W2"], surface["density"]
    z0 = float(CH5_VIEW_BOUNDS[4] if z_floor is None else z_floor)
    z = np.full_like(W1, z0, dtype=float)
    face = _density_facecolors(density, prior_kind=prior_kind)
    face[..., 3] = 0.30
    ch5_plot_belief_surface_with_grid(
        ax3d, W1, W2, z, facecolors=face, zorder=2,
    )


def _ch5_proj_states(active=None, *, shadow_u=0.0, collapse_u=0.0, done=()):
    """Per-axis (shadow_u, collapse_u) for orthogonal marginal projection animation."""
    states = {ax: (1.0, 1.0) for ax in done}
    if active is not None:
        states[str(active)] = (float(shadow_u), float(collapse_u))
    return states


def _ch5_proj_shadow_active(proj_states) -> bool:
    """True while any orthogonal marginal projection is visible."""
    if not proj_states:
        return False
    return any(float(s[0]) > 1e-6 or float(s[1]) > 1e-6 for s in proj_states.values())


def _frame_credible_voxel(
    key,
    *,
    study=None,
    exam=None,
    y=None,
    mass=None,
    prior_kind="gaussian",
    voxel_fill_u=1.0,
    cloud_alpha=1.0,
    proj_states=None,
    interval_box_u=0.0,
    cam_azim_u=0.0,
    cam_spin_deg=None,
    show_map=True,
    map_label="most plausible line",
    hq_elev=False,
    probe=None,
    probe_color=None,
    dark_threshold=False,
    map_shadow_color=None,
    show_colormap=False,
    show_basis_quivers=False,
    quiver_alpha=1.0,
    quiver_newton_colors=False,
    n_ellipsoid_layers=0,
    ellipsoid_reveal_u=1.0,
    hess_slide_u=0.0,
    show_hessian_panel=False,
):
    """Duo: 2D data + 3D HPD voxels. Optional ``probe`` replaces MAP chrome with a wanderer."""
    pack = _PACKS[key]
    if study is None:
        study, exam, y = pack["study"], pack["exam"], pack["y"]
        map_ws, map_we, map_bb = _ch5_map_weights(pack, prior_kind=str(prior_kind).lower())
    else:
        study = np.asarray(study, dtype=np.float64)
        exam = np.asarray(exam, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        map_ws, map_we, map_bb = _ch5_map_weights_arrays(
            study, exam, y,
            prior_kind=str(prior_kind).lower(),
            x0=pack["display_w"],
        )
    xl, yl = ch5_plot_limits(key)
    pk = str(prior_kind).lower()

    vox = _ch5_posterior_3d_voxel_data(study, exam, y, prior_kind=pk, mass=mass)
    if probe is None:
        ws, we, bb = map_ws, map_we, map_bb
        use_probe = False
    else:
        ws, we, bb = float(probe[0]), float(probe[1]), float(probe[2])
        use_probe = True

    fig, ax_data, ax3d, axes_k = _g("ch4_figure_duo_weight3d")()
    leg = _g("legend_linear_equation_values_bold_param")(ws, we, bb, "all")
    want_cmap = bool(show_colormap)
    if use_probe and dark_threshold:
        if want_cmap:
            cmap_kw = _ch5_colormap_panel_kw(xl, yl)
            stg, elg = cmap_kw["sigma_stg"], cmap_kw["sigma_elg"]
            Z = _g("sigmoid")(_g("logits_plane")(ws, we, bb, stg, elg))
            _g("ch3_sigma_contourf")(ax_data, stg, elg, Z, zorder=1)
        _g("draw_dataset")(ax_data, study, exam, y, mask=None)
        _ch5_plot_threshold_line(
            ax_data, ws, we, bb, xl, yl,
            color=str(CH5_CRED_WANDER_THRESHOLD_COLOR),
            linestyle="--",
            linewidth=float(CH5_CRED_WANDER_THRESHOLD_LW),
            alpha=1.0,
            label=leg,
            legend_only_if_missing=True,
        )
        ax_data.set_xlim(*xl)
        ax_data.set_ylim(*yl)
        if ax_data.get_legend_handles_labels()[0]:
            ax_data.legend(loc="upper left", prop={"size": _G["LEGEND_SIZE"]})
        _g("finalize_style_legend_tex")(ax_data)
    else:
        panel_kw = _ch5_colormap_panel_kw(xl, yl) if want_cmap else dict(
            boundary_xlim=xl, boundary_ylim=yl,
        )
        _g("ch3_draw_left_panel")(
            ax_data, ws, we, bb, study, exam, y, leg,
            show_colormap=want_cmap, highlight_mistakes_flag=False,
            show_legend=True, **panel_kw,
        )
    knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
    _g("ch3_draw_knob_row")(
        fig, axes_k, ws, we, bb, "all", knob_rgbs, canvas_sides,
        rot_strip_deg=0.0, strip_scale=1.0,
        knob_rots=_g("ch3_k1_knob_rots_at")(ws, we, bb), knob_scales=[1.0, 1.0, 1.0], ax_data=ax_data,
    )
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = CH5_VIEW_BOUNDS
    _g("_ch3_lik_style_ax3d")(ax3d, dlo1, dhi1, dlo2, dhi2, dlob, dhib)
    spin = float(_g("CH3_LIK_3D_CAM_PATH_ROT") if cam_spin_deg is None else cam_spin_deg)
    _ch5_ct_view_init(
        ax3d,
        cam_azim_u=float(cam_azim_u),
        cam_spin_deg=spin,
        hq_elev=bool(hq_elev),
    )
    # Explicit zorder so the MAP marker stays on top of translucent Laplace shells.
    ax3d.computed_zorder = False

    fill_u = float(voxel_fill_u)
    c_alpha = float(np.clip(cloud_alpha, 0.0, 1.0))
    density = vox["density"]
    # Point cloud grow uses MAP as radial origin so the region itself stays fixed.
    if c_alpha > 1e-4:
        if fill_u >= 1.0:
            ch5_draw_hpd_point_cloud(
                ax3d, vox["w1"], vox["w2"], vox["b"], vox["mask"], density,
                ws=map_ws, we=map_we, bb=map_bb, fill_u=1.0, prior_kind=pk,
                alpha=0.72 * c_alpha,
            )
        elif fill_u > 0.0:
            ch5_draw_hpd_voxels_fill(
                ax3d, vox["w1"], vox["w2"], vox["b"], vox["mask"], density,
                map_ws, map_we, map_bb, fill_u, prior_kind=pk,
                alpha=0.72 * c_alpha,
            )

    intervals = vox["intervals"]
    bounds = vox["bounds"]
    shadow_col = (
        str(map_shadow_color) if map_shadow_color is not None
        else (str(CH5_CRED_WANDER_PROBE_COLOR) if use_probe else "#e04a4a")
    )
    # Wander clips: pin interval lines to MAP; marker follows probe on each axis.
    pin_lines = use_probe
    mark_size = (
        float(CH5_CRED_WANDER_INTERVAL_MARK_SIZE) if use_probe else 54.0
    )
    mark_alpha = (
        float(CH5_CRED_WANDER_INTERVAL_MARK_ALPHA) if use_probe else 0.58
    )
    for axis_key in ("el", "st", "b"):
        state = (proj_states or {}).get(axis_key)
        if state is None:
            continue
        shadow_u, collapse_u = state
        ch5_draw_hpd_orthogonal_projection(
            ax3d, vox["w1"], vox["w2"], vox["b"], vox["mask"],
            intervals, ws, we, bb, bounds, axis_key,
            shadow_u=shadow_u, collapse_u=collapse_u,
            map_shadow_color=shadow_col,
            map_shadow_size=mark_size,
            map_shadow_alpha=mark_alpha,
            line_ws=map_ws if pin_lines else None,
            line_we=map_we if pin_lines else None,
            line_bb=map_bb if pin_lines else None,
        )

    box_u = float(interval_box_u)
    if box_u > 1e-4:
        ch5_draw_interval_parallelepiped(
            ax3d, intervals, reveal_u=box_u, grow_from=(map_ws, map_we, map_bb),
        )

    eigen = None
    q_cols = None
    cell_colors = None
    want_ellipsoids = int(n_ellipsoid_layers) > 0
    q_a = float(np.clip(quiver_alpha, 0.0, 1.0))
    want_quivers = bool(show_basis_quivers) and q_a > 1e-4
    # Defer the marker when shells/quivers are present so one shared origin is drawn last.
    defer_marker = want_ellipsoids or want_quivers
    if use_probe and not defer_marker:
        pcol = str(probe_color if probe_color is not None else CH5_CRED_WANDER_PROBE_COLOR)
        ch5_draw_map_parameter_marker(ax3d, ws, we, bb, color=pcol, edgecolor="white", s=280.0)
    elif (
        bool(show_map) and not defer_marker
        and not _ch5_proj_shadow_active(proj_states) and box_u < 1e-4
    ):
        mcol = _g("FAIL_COLOR")
        ch5_draw_map_parameter_marker(ax3d, ws, we, bb, color=mcol)
        ch5_draw_map_parameter_annotation(
            fig, ax3d, ws, we, bb,
            annotate_fn=_g("_ch3_lik_w12_here_annotation"),
            color=mcol, edgecolor="white", text_color="white",
            label=str(map_label),
        )

    if bool(show_basis_quivers) or bool(show_hessian_panel) or want_ellipsoids:
        eigen = ch5_hessian_eigen_frame(
            study, exam, y, (map_ws, map_we, map_bb),
            prior_kind=pk,
            nll_fn=_g("_ch3_nll_sum_on_flat_grid"),
            bounds=bounds,
        )
        if bool(quiver_newton_colors):
            q_cols = ch5_eigen_quiver_newton_colors(eigen["dirs"])
            axis_active = [True, True, True]
        else:
            q_cols, axis_active = ch5_eigen_quiver_color_plan(
                eigen["dirs"],
                dominance=float(CH5_MAP_QUIVER_DOMINANCE),
                black=str(CH5_MAP_QUIVER_BLACK),
            )
        cell_colors = ch5_hessian_cell_colors_for_axes(
            axis_active, black=None,
        )

    # Visual center follows the on-screen marker (probe if set, else continuous MAP).
    ox, oy, oz = float(ws), float(we), float(bb)

    if want_ellipsoids and eigen is not None:
        base_radii = ch5_laplace_ellipsoid_radii(
            eigen["evals"], eigen["lengths"], scale=1.0,
        )
        scales = list(CH5_ELLIPSOID_LAYER_SCALES)
        face_as = list(CH5_ELLIPSOID_FACE_ALPHAS)
        edge_as = list(CH5_ELLIPSOID_EDGE_ALPHAS)
        n_show = max(0, min(int(n_ellipsoid_layers), len(scales)))
        face_cols = ("#3b82f6", "#60a5fa", "#93c5fd")
        edge_cols = ("#1e3a8a", "#1d4ed8", "#3b82f6")
        for i in range(n_show):
            local_u = 1.0
            if i == n_show - 1:
                local_u = float(np.clip(ellipsoid_reveal_u, 0.0, 1.0))
            if local_u <= 1e-4:
                continue
            sc = float(scales[i]) * (0.35 + 0.65 * local_u)
            fa = float(face_as[i]) * local_u
            ea = float(edge_as[i]) * local_u
            ch5_draw_laplace_ellipsoid(
                ax3d,
                (ox, oy, oz),
                eigen["dirs"],
                base_radii * sc,
                n_u=int(CH5_ELLIPSOID_MESH_U),
                n_v=int(CH5_ELLIPSOID_MESH_V),
                face_color=face_cols[min(i, len(face_cols) - 1)],
                face_alpha=fa,
                edge_color=edge_cols[min(i, len(edge_cols) - 1)],
                edge_alpha=ea,
                zorder=12 + i,
            )

    if want_quivers and eigen is not None:
        ch5_draw_map_basis_quivers(
            ax3d, ox, oy, oz,
            dirs=tuple(eigen["dirs"]),
            lengths=tuple(eigen["lengths"]),
            lw=float(CH5_MAP_QUIVER_LW),
            colors=tuple(q_cols),
            alpha=q_a,
        )

    if defer_marker:
        if use_probe:
            pcol = str(probe_color if probe_color is not None else CH5_CRED_WANDER_PROBE_COLOR)
        else:
            pcol = str(_g("FAIL_COLOR"))
        # Scatter (not a buried sphere mesh) so the point reads at the shared origin.
        ch5_draw_map_parameter_marker(
            ax3d, ox, oy, oz,
            color=pcol, edgecolor="white", s=340.0,
            as_sphere=False,
        )

    slide_pack = _ch5_apply_hessian_side_slide(
        fig, ax_data, axes_k, ax3d, slide_u=float(hess_slide_u),
    )
    if slide_pack is not None and bool(show_hessian_panel):
        ax_h, panel_u = slide_pack
        _ch5_draw_hessian_side_panel(
            ax_h, cell_colors=cell_colors, write_u=float(panel_u),
        )
    return _fig_to_plot(fig)


def _ch5_snap_to_hpd(w1c, w2c, bc, ws, we, bb):
    """Nearest HPD cell center to ``(ws, we, bb)``."""
    if w1c.size == 0:
        return float(ws), float(we), float(bb)
    d2 = (w1c - float(ws)) ** 2 + (w2c - float(we)) ** 2 + (bc - float(bb)) ** 2
    i = int(np.argmin(d2))
    return float(w1c[i]), float(w2c[i]), float(bc[i])


def _ch5_credible_wander_waypoints(vox, map_ws, map_we, map_bb):
    """MAP → interval extremes (snapped into HPD) → MAP."""
    w1c, w2c, bc = _ch5_hpd_cell_centers(vox["w1"], vox["w2"], vox["b"], vox["mask"])
    iv = vox["intervals"]
    st_lo, st_hi = iv["st"]
    el_lo, el_hi = iv["el"]
    b_lo, b_hi = iv["b"]
    map_pt = (float(map_ws), float(map_we), float(map_bb))
    raw = [
        map_pt,
        (st_hi, map_we, map_bb),
        (st_hi, el_hi, map_bb),
        (map_ws, el_hi, b_lo),
        (st_lo, el_lo, b_hi),
        (st_lo, map_we, map_bb),
        map_pt,
    ]
    return [_ch5_snap_to_hpd(w1c, w2c, bc, *p) for p in raw]


def _ch5_point_in_hpd_mask(vox, ws, we, bb, *, tol=None) -> bool:
    """True if ``(ws,we,bb)`` falls in (or within ``tol`` of) an HPD cell."""
    w1c, w2c, bc = _ch5_hpd_cell_centers(vox["w1"], vox["w2"], vox["b"], vox["mask"])
    if w1c.size == 0:
        return False
    d2 = (w1c - float(ws)) ** 2 + (w2c - float(we)) ** 2 + (bc - float(bb)) ** 2
    if tol is None:
        w1 = np.asarray(vox["w1"], dtype=np.float64)
        dw = float(np.min(np.diff(w1))) if w1.size > 1 else 0.2
        tol = 0.75 * dw
    return bool(np.min(d2) <= float(tol) ** 2)


def _ch5_hpd_anchor(vox, map_ws, map_we, map_bb) -> np.ndarray:
    """Nearest HPD cell center to the continuous MAP (path origin inside the set)."""
    w1c, w2c, bc = _ch5_hpd_cell_centers(vox["w1"], vox["w2"], vox["b"], vox["mask"])
    return np.array(
        _ch5_snap_to_hpd(w1c, w2c, bc, map_ws, map_we, map_bb),
        dtype=np.float64,
    )


def _ch5_hpd_ray_alpha_range(vox, map_pt, u) -> tuple[float, float]:
    """Largest [α_lo, α_hi] with MAP + α·û inside the HPD (û unit)."""
    map_pt = np.asarray(map_pt, dtype=np.float64).reshape(3)
    u = np.asarray(u, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(u))
    if nrm < 1e-12:
        return 0.0, 0.0
    u = u / nrm
    w1 = np.asarray(vox["w1"], dtype=np.float64)
    step = 0.35 * float(np.min(np.diff(w1))) if w1.size > 1 else 0.08
    step = max(step, 0.04)

    def _extent(sign: float) -> float:
        last_ok = 0.0
        alpha = 0.0
        for _ in range(400):
            alpha += step
            p = map_pt + float(sign) * alpha * u
            if not _ch5_point_in_hpd_mask(vox, float(p[0]), float(p[1]), float(p[2])):
                return last_ok
            last_ok = alpha
        return last_ok

    return float(-_extent(-1.0)), float(_extent(+1.0))


def _ch5_credible_thickness_axis(vox, map_ws, map_we, map_bb):
    """
    Thickness direction through MAP: unit vector û along the thinnest PCA axis
    of the HPD, plus scalar range [α_lo, α_hi] so MAP + α·û stays in the HPD.

    The probe path is always a constant factor α times the fixed direction û.
    """
    w1c, w2c, bc = _ch5_hpd_cell_centers(vox["w1"], vox["w2"], vox["b"], vox["mask"])
    map_pt = _ch5_hpd_anchor(vox, map_ws, map_we, map_bb)
    if w1c.size < 4:
        return map_pt, np.array([1.0, 0.0, 0.0]), 0.0, 0.0
    C = np.column_stack([w1c, w2c, bc]) - map_pt
    cov = (C.T @ C) / max(float(C.shape[0]), 1.0)
    evals, evecs = np.linalg.eigh(cov)
    u = np.asarray(evecs[:, int(np.argmin(evals))], dtype=np.float64)
    nrm = float(np.linalg.norm(u))
    if nrm < 1e-12:
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        u = u / nrm

    # Prefer projected HPD span on the thin axis (cells near the axis).
    t = C @ u
    resid = np.linalg.norm(C - np.outer(t, u), axis=1)
    w1 = np.asarray(vox["w1"], dtype=np.float64)
    dw = float(np.min(np.diff(w1))) if w1.size > 1 else 0.2
    axis_tol = 1.25 * dw
    keep = resid <= axis_tol
    if int(np.count_nonzero(keep)) < 3:
        keep = resid <= float(np.percentile(resid, 50.0))
    if not np.any(keep):
        keep = np.ones(t.shape[0], dtype=bool)
    a_lo = float(np.min(t[keep]))
    a_hi = float(np.max(t[keep]))

    # Shrink slightly so continuous points stay inside (not just on the rim).
    pad = 0.05 * max(a_hi - a_lo, dw)
    a_lo = min(0.0, a_lo + pad)
    a_hi = max(0.0, a_hi - pad)

    # Verify by walking; pull back if a sample leaves the HPD.
    def _clip(sign: float, a_end: float) -> float:
        if abs(a_end) < 1e-9:
            return 0.0
        n = 48
        last = 0.0
        for tv in np.linspace(0.0, 1.0, n, endpoint=True):
            a = float(a_end) * float(tv)
            p = map_pt + a * u
            if not _ch5_point_in_hpd_mask(vox, float(p[0]), float(p[1]), float(p[2])):
                return last
            last = a
        return last

    a_hi = float(_clip(+1.0, a_hi))
    a_lo = float(_clip(-1.0, a_lo))
    return map_pt, u, a_lo, a_hi


def _ch5_thickness_alpha_path(a_lo, a_hi, *, n_seg):
    """Scalar factors: 0 → +hi → 0 → lo → 0 (out and back both ways)."""
    n = max(int(n_seg), 2)
    segs = [
        np.linspace(0.0, float(a_hi), n, endpoint=True),
        np.linspace(float(a_hi), 0.0, n, endpoint=True),
        np.linspace(0.0, float(a_lo), n, endpoint=True),
        np.linspace(float(a_lo), 0.0, n, endpoint=True),
    ]
    out = [float(segs[0][0])]
    for seg in segs:
        for a in seg[1:]:
            out.append(float(a))
    return out


def _ch5_hpd_radial_scale_range(vox, map_ws, map_we, map_bb, *, step=None):
    """
    Largest [s_lo, s_hi] with s·MAP still inside the HPD.

    Factors step by ``CH5_CRED_THICK_SCALE_STEP`` (default 0.001): grow as
    1.001, 1.002, … and shrink as 0.999, 0.998, … until the scaled point exits.
    """
    step = float(CH5_CRED_THICK_SCALE_STEP if step is None else step)
    step = max(step, 1e-6)
    map_pt = np.array(
        [float(map_ws), float(map_we), float(map_bb)], dtype=np.float64,
    )
    if not _ch5_point_in_hpd_mask(
        vox, float(map_pt[0]), float(map_pt[1]), float(map_pt[2]),
    ):
        map_pt = _ch5_hpd_anchor(vox, map_ws, map_we, map_bb)

    def _extent(sign: float) -> float:
        last = 1.0
        k = 0
        while k < 20000:
            k += 1
            s = 1.0 + float(sign) * k * step
            if s <= 1e-6:
                return last
            p = s * map_pt
            if not _ch5_point_in_hpd_mask(
                vox, float(p[0]), float(p[1]), float(p[2]),
            ):
                return last
            last = float(s)
        return last

    s_hi = float(_extent(+1.0))
    s_lo = float(_extent(-1.0))
    # Keep a hair inside the rim so continuous samples stay in-mask.
    pad = 0.15 * step
    if s_hi > 1.0 + pad:
        s_hi = max(1.0, s_hi - pad)
    if s_lo < 1.0 - pad:
        s_lo = min(1.0, s_lo + pad)
    return s_lo, s_hi, map_pt


def _ch5_radial_scale_path(s_lo, s_hi, *, n_seg):
    """Scale factors: 1 → s_hi → 1 → s_lo → 1."""
    n = max(int(n_seg), 2)
    segs = [
        np.linspace(1.0, float(s_hi), n, endpoint=True),
        np.linspace(float(s_hi), 1.0, n, endpoint=True),
        np.linspace(1.0, float(s_lo), n, endpoint=True),
        np.linspace(float(s_lo), 1.0, n, endpoint=True),
    ]
    out = [float(segs[0][0])]
    for seg in segs:
        for a in seg[1:]:
            out.append(float(a))
    return out


def _ch5_axis_unit_vectors():
    """(name, û) for one-at-a-time w_ST / w_EL / b sweeps."""
    return (
        ("st", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
        ("el", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
        ("b", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
    )


def _build_credible_interval_wander_story(clip_id):
    """
    Per dataset (zoomed duo): credible voxels + intervals → drop MAP chrome →
    black probe with interval face shadows → wander HPD → dark 2D threshold.
    """
    frames: list = []
    hold = int(CH5_CRED_WANDER_N_HOLD)
    n_morph = int(CH5_CRED_WANDER_N_MORPH)
    n_seg = int(CH5_CRED_WANDER_N_SEG)
    pk = "uniform"
    mass = float(CH5_CREDIBLE_MASS)
    orbit_deg = float(CH5_HQ_GRID_ORBIT_DEG)
    smooth = _g("ch3_knob_smoothstep")
    finale_proj = _ch5_proj_states(done=("el", "st", "b"))
    cam = dict(hq_elev=True, cam_azim_u=1.0, cam_spin_deg=orbit_deg)
    red = _g("FAIL_COLOR")
    black = str(CH5_CRED_WANDER_PROBE_COLOR)

    def _cell(key, **kw):
        return _frame_credible_voxel(
            key,
            prior_kind=pk,
            mass=mass,
            voxel_fill_u=1.0,
            proj_states=finale_proj,
            show_map=False,
            show_colormap=True,
            **cam,
            **kw,
        )

    for key in CH5_DATASET_KEYS:
        pack = _PACKS[key]
        map_ws, map_we, map_bb = _ch5_map_weights(pack, prior_kind=pk)
        map_pt = (map_ws, map_we, map_bb)
        vox = _ch5_posterior_3d_voxel_data(
            pack["study"], pack["exam"], pack["y"], prior_kind=pk, mass=mass,
        )
        waypoints = _ch5_credible_wander_waypoints(vox, map_ws, map_we, map_bb)

        # Opening: MAP with label (no intervals yet).
        labeled = _frame_credible_voxel(
            key, prior_kind=pk, mass=mass, voxel_fill_u=1.0,
            proj_states=None, show_map=True, hq_elev=True,
            cam_azim_u=1.0, cam_spin_deg=orbit_deg,
            show_colormap=True,
        )
        frames.extend(_hold(_ch5_finish_duo_export(labeled, clip_id), hold))

        # Intervals on; red MAP face-shadows; label gone.
        red_probe = _cell(
            key, probe=map_pt, probe_color=red, dark_threshold=False,
            map_shadow_color="#e04a4a",
        )
        for tv in np.linspace(0.0, 1.0, n_morph, endpoint=True):
            u = float(smooth(float(tv)))
            frames.append(_ch5_finish_duo_export(
                ch5_crossfade_images(labeled, red_probe, u), clip_id,
            ))
        frames.extend(_hold(_ch5_finish_duo_export(red_probe, clip_id), hold))

        # Morph red → black probe; 2D threshold goes dark.
        black_probe = _cell(
            key, probe=map_pt, probe_color=black, dark_threshold=True,
            map_shadow_color=black,
        )
        for tv in np.linspace(0.0, 1.0, n_morph, endpoint=True):
            u = float(smooth(float(tv)))
            frames.append(_ch5_finish_duo_export(
                ch5_crossfade_images(red_probe, black_probe, u), clip_id,
            ))
        frames.extend(_hold(_ch5_finish_duo_export(black_probe, clip_id), hold))

        # Wander through HPD waypoints (ends at MAP).
        for i in range(len(waypoints) - 1):
            p0, p1 = waypoints[i], waypoints[i + 1]
            for tv in np.linspace(0.0, 1.0, n_seg, endpoint=True):
                u = float(smooth(float(tv)))
                pt = (
                    (1.0 - u) * p0[0] + u * p1[0],
                    (1.0 - u) * p0[1] + u * p1[1],
                    (1.0 - u) * p0[2] + u * p1[2],
                )
                frames.append(_ch5_finish_duo_export(
                    _cell(
                        key, probe=pt, probe_color=black, dark_threshold=True,
                        map_shadow_color=black,
                    ),
                    clip_id,
                ))
            frames.extend(_hold(frames[-1], max(2, hold // 2)))

        frames.extend(_hold(_ch5_finish_duo_export(black_probe, clip_id), hold))

    frames.extend(_hold(frames[-1], hold))
    return frames


def _ch5_cam_from_spin(spin_deg) -> tuple[float, float]:
    """(cam_azim_u, cam_spin_deg) for an absolute azim offset from the og view.

    Positive ``spin_deg`` is counter-clockwise (azim increases); negative is CW.
    """
    s = float(spin_deg)
    if abs(s) < 1e-9:
        return 0.0, 0.0
    return 1.0, s


def _build_credible_thickness_wander_story(
    clip_id, *, view_spins=None, dataset_keys=None,
):
    """
    Like ch5_61, then radial scale wander (s·MAP) at one or more camera angles.

    ``view_spins`` — absolute azim offsets from the og view (+ = CCW). Default
    ``[90]`` is ch5_62. Pass ``[0, 45]`` to wander at 0° then 45°.
    """
    frames: list = []
    hold = int(CH5_CRED_WANDER_N_HOLD)
    n_morph = int(CH5_CRED_WANDER_N_MORPH)
    n_rot = int(CH5_CRED_THICK_N_ROT)
    n_seg = int(CH5_CRED_THICK_N_SEG)
    spins = (
        [float(CH5_CRED_THICK_ROT_DEG)] if view_spins is None
        else [float(s) for s in view_spins]
    )
    keys = list(CH5_DATASET_KEYS if dataset_keys is None else dataset_keys)
    pk = "uniform"
    mass = float(CH5_CREDIBLE_MASS)
    smooth = _g("ch3_knob_smoothstep")
    finale_proj = _ch5_proj_states(done=("el", "st", "b"))
    red = _g("FAIL_COLOR")
    black = str(CH5_CRED_WANDER_PROBE_COLOR)

    def _cell(key, *, cam_azim_u=0.0, cam_spin_deg=0.0, **kw):
        return _frame_credible_voxel(
            key,
            prior_kind=pk,
            mass=mass,
            voxel_fill_u=1.0,
            proj_states=finale_proj,
            show_map=False,
            show_colormap=True,
            hq_elev=True,
            cam_azim_u=float(cam_azim_u),
            cam_spin_deg=float(cam_spin_deg),
            **kw,
        )

    def _goto_spin(key, spin0, spin1, probe):
        if abs(float(spin1) - float(spin0)) < 1e-6:
            return
        for tv in np.linspace(0.0, 1.0, n_rot, endpoint=True):
            u = float(smooth(float(tv)))
            spin = (1.0 - u) * float(spin0) + u * float(spin1)
            cu, cs = _ch5_cam_from_spin(spin)
            frames.append(_ch5_finish_duo_export(
                _cell(
                    key, cam_azim_u=cu, cam_spin_deg=cs,
                    probe=probe, probe_color=black, dark_threshold=True,
                    map_shadow_color=black,
                ),
                clip_id,
            ))

    for key in keys:
        pack = _PACKS[key]
        map_ws, map_we, map_bb = _ch5_map_weights(pack, prior_kind=pk)
        map_pt = (map_ws, map_we, map_bb)
        vox = _ch5_posterior_3d_voxel_data(
            pack["study"], pack["exam"], pack["y"], prior_kind=pk, mass=mass,
        )
        s_lo, s_hi, map_vec = _ch5_hpd_radial_scale_range(
            vox, map_ws, map_we, map_bb,
        )
        scales = _ch5_radial_scale_path(s_lo, s_hi, n_seg=n_seg)
        scale_pt = (float(map_vec[0]), float(map_vec[1]), float(map_vec[2]))

        labeled = _frame_credible_voxel(
            key, prior_kind=pk, mass=mass, voxel_fill_u=1.0,
            proj_states=None, show_map=True, hq_elev=True,
            cam_azim_u=0.0, cam_spin_deg=0.0, show_colormap=True,
        )
        frames.extend(_hold(_ch5_finish_duo_export(labeled, clip_id), hold))

        red_probe = _cell(
            key, cam_azim_u=0.0, cam_spin_deg=0.0,
            probe=map_pt, probe_color=red, dark_threshold=False,
            map_shadow_color="#e04a4a",
        )
        for tv in np.linspace(0.0, 1.0, n_morph, endpoint=True):
            u = float(smooth(float(tv)))
            frames.append(_ch5_finish_duo_export(
                ch5_crossfade_images(labeled, red_probe, u), clip_id,
            ))
        frames.extend(_hold(_ch5_finish_duo_export(red_probe, clip_id), hold))

        black_probe0 = _cell(
            key, cam_azim_u=0.0, cam_spin_deg=0.0,
            probe=map_pt, probe_color=black, dark_threshold=True,
            map_shadow_color=black,
        )
        for tv in np.linspace(0.0, 1.0, n_morph, endpoint=True):
            u = float(smooth(float(tv)))
            frames.append(_ch5_finish_duo_export(
                ch5_crossfade_images(red_probe, black_probe0, u), clip_id,
            ))
        frames.extend(_hold(_ch5_finish_duo_export(black_probe0, clip_id), hold))

        cur_spin = 0.0
        for spin_target in spins:
            _goto_spin(key, cur_spin, spin_target, scale_pt)
            cur_spin = float(spin_target)
            cu, cs = _ch5_cam_from_spin(cur_spin)
            black_probe = _cell(
                key, cam_azim_u=cu, cam_spin_deg=cs,
                probe=scale_pt, probe_color=black, dark_threshold=True,
                map_shadow_color=black,
            )
            frames.extend(_hold(_ch5_finish_duo_export(black_probe, clip_id), hold))

            for s in scales:
                p = float(s) * map_vec
                pt = (float(p[0]), float(p[1]), float(p[2]))
                frames.append(_ch5_finish_duo_export(
                    _cell(
                        key, cam_azim_u=cu, cam_spin_deg=cs,
                        probe=pt, probe_color=black, dark_threshold=True,
                        map_shadow_color=black,
                    ),
                    clip_id,
                ))
            frames.extend(_hold(_ch5_finish_duo_export(black_probe, clip_id), hold))

    frames.extend(_hold(frames[-1], hold))
    return frames


def _build_credible_axis_wander_story(
    clip_id, *, dataset_keys=None, axis_view_spins=None, setup_spin=None,
):
    """
    Intervals → black probe, then vary w_ST / w_EL / b one at a time inside the HPD.

    Default (ch5_63): all datasets, 90° CCW setup, all sweeps at that view.

    D1 camera variant: pass ``dataset_keys=["D1"]``, ``setup_spin=0``, and
    ``axis_view_spins={"st": +45, "el": -45, "b": 0}`` (+ = CCW from og view).
    """
    frames: list = []
    hold = int(CH5_CRED_WANDER_N_HOLD)
    n_morph = int(CH5_CRED_WANDER_N_MORPH)
    n_rot = int(CH5_CRED_THICK_N_ROT)
    n_seg = int(CH5_CRED_AXIS_N_SEG)
    axis_hold = int(CH5_CRED_AXIS_N_HOLD)
    default_spin = float(CH5_CRED_THICK_ROT_DEG)
    setup = float(default_spin if setup_spin is None else setup_spin)
    keys = list(CH5_DATASET_KEYS if dataset_keys is None else dataset_keys)
    # Absolute azim offset per axis name; None → stay at setup spin.
    axis_spins = None if axis_view_spins is None else {
        str(k): float(v) for k, v in dict(axis_view_spins).items()
    }
    pk = "uniform"
    mass = float(CH5_CREDIBLE_MASS)
    smooth = _g("ch3_knob_smoothstep")
    finale_proj = _ch5_proj_states(done=("el", "st", "b"))
    red = _g("FAIL_COLOR")
    black = str(CH5_CRED_WANDER_PROBE_COLOR)

    def _cell(key, *, cam_azim_u=0.0, cam_spin_deg=0.0, **kw):
        return _frame_credible_voxel(
            key,
            prior_kind=pk,
            mass=mass,
            voxel_fill_u=1.0,
            proj_states=finale_proj,
            show_map=False,
            show_colormap=True,
            hq_elev=True,
            cam_azim_u=float(cam_azim_u),
            cam_spin_deg=float(cam_spin_deg),
            **kw,
        )

    def _goto_spin(key, spin0, spin1, probe):
        if abs(float(spin1) - float(spin0)) < 1e-6:
            return
        for tv in np.linspace(0.0, 1.0, n_rot, endpoint=True):
            u = float(smooth(float(tv)))
            spin = (1.0 - u) * float(spin0) + u * float(spin1)
            cu, cs = _ch5_cam_from_spin(spin)
            frames.append(_ch5_finish_duo_export(
                _cell(
                    key, cam_azim_u=cu, cam_spin_deg=cs,
                    probe=probe, probe_color=black, dark_threshold=True,
                    map_shadow_color=black,
                ),
                clip_id,
            ))

    for key in keys:
        pack = _PACKS[key]
        map_ws, map_we, map_bb = _ch5_map_weights(pack, prior_kind=pk)
        map_pt = (map_ws, map_we, map_bb)
        vox = _ch5_posterior_3d_voxel_data(
            pack["study"], pack["exam"], pack["y"], prior_kind=pk, mass=mass,
        )
        map_vec = _ch5_hpd_anchor(vox, map_ws, map_we, map_bb)
        anchor = (float(map_vec[0]), float(map_vec[1]), float(map_vec[2]))

        labeled = _frame_credible_voxel(
            key, prior_kind=pk, mass=mass, voxel_fill_u=1.0,
            proj_states=None, show_map=True, hq_elev=True,
            cam_azim_u=0.0, cam_spin_deg=0.0, show_colormap=True,
        )
        frames.extend(_hold(_ch5_finish_duo_export(labeled, clip_id), hold))

        red_probe = _cell(
            key, cam_azim_u=0.0, cam_spin_deg=0.0,
            probe=map_pt, probe_color=red, dark_threshold=False,
            map_shadow_color="#e04a4a",
        )
        for tv in np.linspace(0.0, 1.0, n_morph, endpoint=True):
            u = float(smooth(float(tv)))
            frames.append(_ch5_finish_duo_export(
                ch5_crossfade_images(labeled, red_probe, u), clip_id,
            ))
        frames.extend(_hold(_ch5_finish_duo_export(red_probe, clip_id), hold))

        black_probe0 = _cell(
            key, cam_azim_u=0.0, cam_spin_deg=0.0,
            probe=map_pt, probe_color=black, dark_threshold=True,
            map_shadow_color=black,
        )
        for tv in np.linspace(0.0, 1.0, n_morph, endpoint=True):
            u = float(smooth(float(tv)))
            frames.append(_ch5_finish_duo_export(
                ch5_crossfade_images(red_probe, black_probe0, u), clip_id,
            ))
        frames.extend(_hold(_ch5_finish_duo_export(black_probe0, clip_id), hold))

        cur_spin = 0.0
        _goto_spin(key, cur_spin, setup, map_pt)
        cur_spin = float(setup)
        cu0, cs0 = _ch5_cam_from_spin(cur_spin)
        black_probe = _cell(
            key, cam_azim_u=cu0, cam_spin_deg=cs0,
            probe=anchor, probe_color=black, dark_threshold=True,
            map_shadow_color=black,
        )
        frames.extend(_hold(_ch5_finish_duo_export(black_probe, clip_id), hold))

        for name, e_i in _ch5_axis_unit_vectors():
            target = (
                float(axis_spins[name]) if axis_spins is not None and name in axis_spins
                else float(cur_spin if axis_spins is None else setup)
            )
            if axis_spins is not None:
                _goto_spin(key, cur_spin, target, anchor)
                cur_spin = float(target)
            cu, cs = _ch5_cam_from_spin(cur_spin)
            black_probe = _cell(
                key, cam_azim_u=cu, cam_spin_deg=cs,
                probe=anchor, probe_color=black, dark_threshold=True,
                map_shadow_color=black,
            )
            frames.extend(_hold(_ch5_finish_duo_export(black_probe, clip_id), axis_hold))

            a_lo, a_hi = _ch5_hpd_ray_alpha_range(vox, map_vec, e_i)
            for alpha in _ch5_thickness_alpha_path(a_lo, a_hi, n_seg=n_seg):
                p = map_vec + float(alpha) * e_i
                pt = (float(p[0]), float(p[1]), float(p[2]))
                frames.append(_ch5_finish_duo_export(
                    _cell(
                        key, cam_azim_u=cu, cam_spin_deg=cs,
                        probe=pt, probe_color=black, dark_threshold=True,
                        map_shadow_color=black,
                    ),
                    clip_id,
                ))
            frames.extend(_hold(
                _ch5_finish_duo_export(black_probe, clip_id), axis_hold,
            ))

        frames.extend(_hold(_ch5_finish_duo_export(black_probe, clip_id), hold))

    frames.extend(_hold(frames[-1], hold))
    return frames


def _frame_credible_voxel_grid_2x2(*, mass=None, active_keys=None, prior_kind="gaussian", **kw):
    """2×2 grid of 3D credible voxel cutouts in parameter space."""
    keys = list(CH5_DATASET_KEYS if active_keys is None else active_keys)
    cells = [[None, None], [None, None]]
    for key in keys:
        i, j = _CH5_QUAD_SLOTS[key]
        cells[i][j] = _frame_credible_voxel(key, mass=mass, prior_kind=prior_kind, **kw)
    return ch5_composite_2x2_quadrants(cells)


def _ch5_voxel_tour_grid_cells(
    *,
    prior_kind="uniform",
    voxel_fill_u=1.0,
    show_map=True,
    hq_elev=True,
    cam_azim_u=0.0,
    cam_spin_deg=0.0,
    mass=None,
):
    """2×2 list-of-lists of voxel duo frames (for focus/zoom compositing)."""
    cells = [[None, None], [None, None]]
    for key in CH5_DATASET_KEYS:
        i, j = _CH5_QUAD_SLOTS[key]
        cells[i][j] = _frame_credible_voxel(
            key,
            prior_kind=prior_kind,
            mass=mass,
            voxel_fill_u=float(voxel_fill_u),
            show_map=bool(show_map),
            hq_elev=bool(hq_elev),
            cam_azim_u=float(cam_azim_u),
            cam_spin_deg=float(cam_spin_deg),
            map_label="most plausible line",
        )
    return cells


def _build_ct_to_voxel_grid_tour_story(clip_id):
    """
    Continue from ch5_29 end: fade CT planes → MAP in 3D → grow 95% voxels →
    per dataset: emphasize/zoom → 360° → credible shadows/intervals → drop voxels →
    interval parallelepiped → 360° → voxels back → zoom out.
    """
    frames: list = []
    hold = int(CH5_CT_VOXEL_TOUR_N_HOLD)
    n_erase = int(CH5_CT_VOXEL_TOUR_N_ERASE)
    n_map = int(CH5_CT_VOXEL_TOUR_N_MAP)
    n_fill = int(CH5_VOXEL_N_FILL)
    n_fade = int(CH5_HQ_GRID_N_FOCUS_FADE)
    n_zoom = int(CH5_HQ_GRID_N_ZOOM)
    n_zhold = int(CH5_HQ_GRID_N_ZOOM_HOLD)
    n_orbit = int(CH5_HQ_GRID_N_ZOOM_ORBIT)
    n_shadow = int(CH5_VOXEL_N_PROJ_SHADOW)
    n_collapse = int(CH5_VOXEL_N_PROJ_COLLAPSE)
    n_box = int(CH5_CT_VOXEL_TOUR_N_BOX)
    n_swap = int(CH5_CT_VOXEL_TOUR_N_VOXEL_SWAP)
    proj_hold = max(2, hold // 2)
    orbit_deg = float(CH5_HQ_GRID_ORBIT_DEG)
    focus_kw = dict(
        grey_weight=float(CH5_HQ_GRID_FOCUS_DIM_GREY),
        alpha_min=float(CH5_HQ_GRID_FOCUS_DIM_ALPHA),
    )
    pk = "uniform"
    keys = list(CH5_DATASET_KEYS)
    b_hi = float(CH5_VIEW_BOUNDS[5])
    smooth = _g("ch3_knob_smoothstep")
    ct_kw = dict(
        prior_kind=pk,
        ct_gn=int(CH5_HQ_CT_GRID),
        show_map=True,
        show_threshold=True,
        hq_elev=True,
    )
    vox_base = dict(
        prior_kind=pk,
        mass=float(CH5_CREDIBLE_MASS),
        hq_elev=True,
        cam_azim_u=0.0,
        cam_spin_deg=0.0,
        map_label="most plausible line",
    )
    mass = float(CH5_CREDIBLE_MASS)

    def _cell(
        key, *,
        voxel_fill_u=1.0,
        proj_states=None,
        interval_box_u=0.0,
        cam_azim_u=1.0,
        cam_spin_deg=None,
        show_map=False,
    ):
        spin = float(orbit_deg if cam_spin_deg is None else cam_spin_deg)
        return _frame_credible_voxel(
            key,
            prior_kind=pk,
            mass=mass,
            hq_elev=True,
            voxel_fill_u=float(voxel_fill_u),
            proj_states=proj_states,
            interval_box_u=float(interval_box_u),
            cam_azim_u=float(cam_azim_u),
            cam_spin_deg=spin,
            show_map=bool(show_map),
            map_label="most plausible line",
        )

    # 1) Hold ch5_29 end pose (b = +3 CT plane).
    ct_end = _frame_ct_grid_2x2(keys, sweep_axis="b", plane_val=b_hi, **ct_kw)
    frames.extend(_hold(_ch5_finish_duo_export(ct_end, clip_id), hold))

    # 2) Simultaneously remove CT planes → empty param axes (no MAP yet).
    empty = _frame_credible_voxel_grid_2x2(
        show_map=False, voxel_fill_u=0.0, **vox_base,
    )
    for tv in np.linspace(0.0, 1.0, n_erase, endpoint=True):
        u = float(smooth(float(tv)))
        frames.append(_ch5_finish_duo_export(
            ch5_crossfade_images(ct_end, empty, u), clip_id,
        ))
    frames.extend(_hold(_ch5_finish_duo_export(empty, clip_id), hold))

    # 3) Add full-space MAP point + “most plausible line” callout.
    map0 = _frame_credible_voxel_grid_2x2(
        show_map=True, voxel_fill_u=0.0, **vox_base,
    )
    for tv in np.linspace(0.0, 1.0, n_map, endpoint=True):
        u = float(smooth(float(tv)))
        frames.append(_ch5_finish_duo_export(
            ch5_crossfade_images(empty, map0, u), clip_id,
        ))
    frames.extend(_hold(_ch5_finish_duo_export(map0, clip_id), hold))

    # 4) Grow 95% credible voxels on the 2×2 grid.
    for tv in np.linspace(0.0, 1.0, n_fill, endpoint=True):
        u = float(smooth(float(tv)))
        grid = _frame_credible_voxel_grid_2x2(
            show_map=True, voxel_fill_u=float(u), **vox_base,
        )
        frames.append(_ch5_finish_duo_export(grid, clip_id))
    cells = _ch5_voxel_tour_grid_cells(
        prior_kind=pk, voxel_fill_u=1.0, show_map=True, hq_elev=True,
        cam_azim_u=0.0, cam_spin_deg=0.0, mass=mass,
    )
    grid_full = ch5_composite_2x2_quadrants(cells)
    frames.extend(_hold(_ch5_finish_duo_export(grid_full, clip_id), hold))

    # 5) Per-dataset: focus → zoom → orbit → shadows/intervals → box → orbit → voxels → out.
    for key in CH5_DATASET_KEYS:
        row, col = _CH5_QUAD_SLOTS[key]
        for tv in np.linspace(0.0, 1.0, n_fade, endpoint=True):
            frames.append(_ch5_finish_duo_export(
                ch5_composite_2x2_focus(
                    cells, key, prev_focus=None, transition_u=float(tv), **focus_kw,
                ),
                clip_id,
            ))
        grid_focus = ch5_composite_2x2_focus(cells, key, dim_u=1.0, **focus_kw)
        frames.extend(_hold(_ch5_finish_duo_export(grid_focus, clip_id), max(1, n_zhold // 2)))
        for tv in np.linspace(0.0, 1.0, n_zoom, endpoint=True):
            frames.append(_ch5_finish_duo_export(
                ch5_quadrant_zoom_frame(grid_focus, row, col, float(tv)), clip_id,
            ))
        cell_ref = _frame_credible_voxel(
            key, voxel_fill_u=1.0, show_map=True, **vox_base,
        )
        frames.extend(_hold(_ch5_finish_duo_export(cell_ref, clip_id), n_zhold))

        # First 360° with voxels + MAP.
        for tv in np.linspace(0.0, 1.0, n_orbit, endpoint=True):
            frames.append(_ch5_finish_duo_export(
                _cell(
                    key, voxel_fill_u=1.0, show_map=True,
                    cam_azim_u=float(tv), cam_spin_deg=orbit_deg,
                ),
                clip_id,
            ))
        frames.extend(_hold(
            _ch5_finish_duo_export(
                _cell(key, voxel_fill_u=1.0, show_map=True, cam_azim_u=1.0),
                clip_id,
            ),
            proj_hold,
        ))

        # Credible face shadows → collapse to marginal intervals (same as voxel clips).
        done = ()
        for axis in ("el", "st", "b"):
            for tv in np.linspace(0.0, 1.0, n_shadow, endpoint=True):
                u = float(smooth(float(tv)))
                frames.append(_ch5_finish_duo_export(
                    _cell(
                        key, voxel_fill_u=1.0,
                        proj_states=_ch5_proj_states(
                            axis, shadow_u=u, collapse_u=0.0, done=done,
                        ),
                    ),
                    clip_id,
                ))
            frames.extend(_hold(
                _ch5_finish_duo_export(
                    _cell(
                        key, voxel_fill_u=1.0,
                        proj_states=_ch5_proj_states(
                            axis, shadow_u=1.0, collapse_u=0.0, done=done,
                        ),
                    ),
                    clip_id,
                ),
                proj_hold,
            ))
            for tv in np.linspace(0.0, 1.0, n_collapse, endpoint=True):
                u = float(smooth(float(tv)))
                frames.append(_ch5_finish_duo_export(
                    _cell(
                        key, voxel_fill_u=1.0,
                        proj_states=_ch5_proj_states(
                            axis, shadow_u=1.0, collapse_u=u, done=done,
                        ),
                    ),
                    clip_id,
                ))
            done = done + (axis,)
            frames.extend(_hold(
                _ch5_finish_duo_export(
                    _cell(
                        key, voxel_fill_u=1.0,
                        proj_states=_ch5_proj_states(done=done),
                    ),
                    clip_id,
                ),
                hold,
            ))

        finale_proj = _ch5_proj_states(done=done)

        # Drop voxel cloud completely (intervals stay).
        for tv in np.linspace(1.0, 0.0, n_swap, endpoint=True):
            u = float(smooth(float(tv)))
            frames.append(_ch5_finish_duo_export(
                _cell(
                    key, voxel_fill_u=float(u),
                    proj_states=finale_proj,
                ),
                clip_id,
            ))
        frames.extend(_hold(
            _ch5_finish_duo_export(
                _cell(key, voxel_fill_u=0.0, proj_states=finale_proj),
                clip_id,
            ),
            proj_hold,
        ))

        # Grow interval parallelepiped from MAP.
        for tv in np.linspace(0.0, 1.0, n_box, endpoint=True):
            u = float(smooth(float(tv)))
            frames.append(_ch5_finish_duo_export(
                _cell(
                    key, voxel_fill_u=0.0,
                    proj_states=finale_proj,
                    interval_box_u=float(u),
                ),
                clip_id,
            ))
        frames.extend(_hold(
            _ch5_finish_duo_export(
                _cell(
                    key, voxel_fill_u=0.0,
                    proj_states=finale_proj,
                    interval_box_u=1.0,
                ),
                clip_id,
            ),
            hold,
        ))

        # Second 360° (continuous: spin=720°, azim_u 0.5→1.0).
        for tv in np.linspace(0.0, 1.0, n_orbit, endpoint=True):
            u = 0.5 + 0.5 * float(tv)
            frames.append(_ch5_finish_duo_export(
                _cell(
                    key, voxel_fill_u=0.0,
                    proj_states=finale_proj,
                    interval_box_u=1.0,
                    cam_azim_u=float(u),
                    cam_spin_deg=2.0 * orbit_deg,
                ),
                clip_id,
            ))
        frames.extend(_hold(
            _ch5_finish_duo_export(
                _cell(
                    key, voxel_fill_u=0.0,
                    proj_states=finale_proj,
                    interval_box_u=1.0,
                    cam_azim_u=1.0,
                    cam_spin_deg=2.0 * orbit_deg,
                ),
                clip_id,
            ),
            proj_hold,
        ))

        # Remove box, restore voxels (intervals stay).
        for tv in np.linspace(1.0, 0.0, n_box, endpoint=True):
            u = float(smooth(float(tv)))
            frames.append(_ch5_finish_duo_export(
                _cell(
                    key, voxel_fill_u=0.0,
                    proj_states=finale_proj,
                    interval_box_u=float(u),
                    cam_azim_u=1.0,
                    cam_spin_deg=2.0 * orbit_deg,
                ),
                clip_id,
            ))
        for tv in np.linspace(0.0, 1.0, n_swap, endpoint=True):
            u = float(smooth(float(tv)))
            frames.append(_ch5_finish_duo_export(
                _cell(
                    key, voxel_fill_u=float(u),
                    proj_states=finale_proj,
                    interval_box_u=0.0,
                    cam_azim_u=1.0,
                    cam_spin_deg=2.0 * orbit_deg,
                ),
                clip_id,
            ))
        frames.extend(_hold(
            _ch5_finish_duo_export(
                _cell(
                    key, voxel_fill_u=1.0,
                    proj_states=finale_proj,
                    cam_azim_u=1.0,
                    cam_spin_deg=2.0 * orbit_deg,
                ),
                clip_id,
            ),
            n_zhold,
        ))

        for tv in np.linspace(1.0, 0.0, n_zoom, endpoint=True):
            frames.append(_ch5_finish_duo_export(
                ch5_quadrant_zoom_frame(grid_full, row, col, float(tv)), clip_id,
            ))

    frames.extend(_hold(frames[-1], hold))
    return frames


def build_ch5_ct_to_voxel_grid_tour(clip_id):
    return _build_ct_to_voxel_grid_tour_story(clip_id)


def build_ch5_credible_interval_wander(clip_id):
    return _build_credible_interval_wander_story(clip_id)


def build_ch5_credible_thickness_wander(clip_id):
    """ch5_62: radial scale wander at 90° CCW from og view."""
    return _build_credible_thickness_wander_story(clip_id, view_spins=(90.0,))


def build_ch5_credible_thickness_wander_az0_45(clip_id):
    """Radial scale wander at 0°, then 45° CCW (62 is the 90° version)."""
    return _build_credible_thickness_wander_story(clip_id, view_spins=(0.0, 45.0))


def build_ch5_credible_axis_wander(clip_id):
    """ch5_63: all datasets, 90° setup, axis sweeps at that view."""
    return _build_credible_axis_wander_story(clip_id)


def build_ch5_credible_axis_wander_D1_views(clip_id):
    """D1 only: w_ST at +45° CCW, w_EL at −45° (CW), b at og (0°)."""
    return _build_credible_axis_wander_story(
        clip_id,
        dataset_keys=("D1",),
        setup_spin=0.0,
        axis_view_spins={"st": 45.0, "el": -45.0, "b": 0.0},
    )


def _build_credible_map_basis_quiver_orbit_story(clip_id):
    """
    D1 credible region: slide 2D off / 3D left (aspect preserved) + side Hessian,
    then eigen-quivers (ST/EL/b accents) + matching Hessian diagonals + 360° orbit.
    """
    frames: list = []
    hold = int(CH5_MAP_QUIVER_N_HOLD)
    n_orbit = int(CH5_MAP_QUIVER_N_ORBIT)
    n_slide = int(CH5_MAP_QUIVER_N_SLIDE)
    pk = "uniform"
    mass = float(CH5_CREDIBLE_MASS)
    key = "D1"
    finale_proj = _ch5_proj_states(done=("el", "st", "b"))

    def _cell(
        *,
        cam_azim_u=0.0,
        cam_spin_deg=0.0,
        show_map=True,
        show_basis_quivers=False,
        hess_slide_u=0.0,
        show_hessian_panel=False,
    ):
        return _frame_credible_voxel(
            key,
            prior_kind=pk,
            mass=mass,
            voxel_fill_u=1.0,
            proj_states=finale_proj,
            show_map=bool(show_map),
            show_colormap=True,
            hq_elev=True,
            cam_azim_u=float(cam_azim_u),
            cam_spin_deg=float(cam_spin_deg),
            show_basis_quivers=bool(show_basis_quivers),
            hess_slide_u=float(hess_slide_u),
            show_hessian_panel=bool(show_hessian_panel),
        )

    def _finish_frame(img):
        return _ch5_finish_duo_export(img, clip_id)

    # Opening: duo intervals + MAP.
    open_img = _cell(show_basis_quivers=False, hess_slide_u=0.0, show_hessian_panel=False)
    frames.extend(_hold(_finish_frame(open_img), hold))

    # Slide 2D off-screen; 3D keeps its box size and moves left; Hessian panel opens.
    for tv in np.linspace(0.0, 1.0, n_slide, endpoint=True):
        frames.append(_finish_frame(_cell(
            show_basis_quivers=False,
            hess_slide_u=float(tv),
            show_hessian_panel=True,
        )))

    # Eigen-quivers appear with ST/EL/b accents matched to Hessian diagonals.
    quiver_img = _cell(
        show_basis_quivers=True, hess_slide_u=1.0, show_hessian_panel=True,
    )
    frames.extend(_hold(_finish_frame(quiver_img), hold))

    # Full 360° CCW orbit.
    for tv in np.linspace(0.0, 1.0, n_orbit, endpoint=True):
        frames.append(_finish_frame(_cell(
            cam_azim_u=float(tv),
            cam_spin_deg=360.0,
            show_basis_quivers=True,
            hess_slide_u=1.0,
            show_hessian_panel=True,
        )))
    frames.extend(_hold(frames[-1], hold))
    return frames


def build_ch5_credible_map_basis_quiver_orbit(clip_id):
    return _build_credible_map_basis_quiver_orbit_story(clip_id)


def _frame_laplace_ellipsoid_preview(
    key="D1",
    *,
    prior_kind="uniform",
    cam_azim_u=0.0,
    cam_spin_deg=0.0,
    n_layers_show=0,
    layer_reveal_u=1.0,
    show_quivers=True,
    hq_elev=True,
):
    """2D data + empty 3D: MAP + optional eigen-quivers + nested Laplace ellipsoids.

    No HPD voxels, no interval shadows, no Hessian text panel.
    """
    pack = _PACKS[key]
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    pk = str(prior_kind).lower()
    map_ws, map_we, map_bb = _ch5_map_weights(pack, prior_kind=pk)
    xl, yl = ch5_plot_limits(key)

    fig, ax_data, ax3d, axes_k = _g("ch4_figure_duo_weight3d")()
    leg = _g("legend_linear_equation_values_bold_param")(map_ws, map_we, map_bb, "all")
    panel_kw = _ch5_colormap_panel_kw(xl, yl)
    _g("ch3_draw_left_panel")(
        ax_data, map_ws, map_we, map_bb, study, exam, y, leg,
        show_colormap=True, highlight_mistakes_flag=False,
        show_legend=True, **panel_kw,
    )
    knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
    _g("ch3_draw_knob_row")(
        fig, axes_k, map_ws, map_we, map_bb, "all", knob_rgbs, canvas_sides,
        rot_strip_deg=0.0, strip_scale=1.0,
        knob_rots=_g("ch3_k1_knob_rots_at")(map_ws, map_we, map_bb),
        knob_scales=[1.0, 1.0, 1.0], ax_data=ax_data,
    )
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = CH5_VIEW_BOUNDS
    _g("_ch3_lik_style_ax3d")(ax3d, dlo1, dhi1, dlo2, dhi2, dlob, dhib)
    spin = float(_g("CH3_LIK_3D_CAM_PATH_ROT") if cam_spin_deg is None else cam_spin_deg)
    _ch5_ct_view_init(
        ax3d,
        cam_azim_u=float(cam_azim_u),
        cam_spin_deg=spin,
        hq_elev=bool(hq_elev),
    )

    eigen = ch5_hessian_eigen_frame(
        study, exam, y, (map_ws, map_we, map_bb),
        prior_kind=pk,
        nll_fn=_g("_ch3_nll_sum_on_flat_grid"),
        bounds=CH5_VIEW_BOUNDS,
    )
    q_cols, _ = ch5_eigen_quiver_color_plan(
        eigen["dirs"],
        dominance=float(CH5_MAP_QUIVER_DOMINANCE),
        black=str(CH5_MAP_QUIVER_BLACK),
    )
    if bool(show_quivers):
        ch5_draw_map_basis_quivers(
            ax3d, map_ws, map_we, map_bb,
            dirs=tuple(eigen["dirs"]),
            lengths=tuple(eigen["lengths"]),
            lw=float(CH5_MAP_QUIVER_LW),
            colors=tuple(q_cols),
        )
    ch5_draw_map_parameter_marker(
        ax3d, map_ws, map_we, map_bb,
        color=str(_g("FAIL_COLOR")), edgecolor="white", s=300.0,
    )

    base_radii = ch5_laplace_ellipsoid_radii(eigen["evals"], eigen["lengths"], scale=1.0)
    scales = list(CH5_ELLIPSOID_LAYER_SCALES)
    face_as = list(CH5_ELLIPSOID_FACE_ALPHAS)
    edge_as = list(CH5_ELLIPSOID_EDGE_ALPHAS)
    n_show = max(0, min(int(n_layers_show), len(scales)))
    # Colors: inner → outer, cooler → softer blue.
    face_cols = ("#3b82f6", "#60a5fa", "#93c5fd")
    edge_cols = ("#1e3a8a", "#1d4ed8", "#3b82f6")
    for i in range(n_show):
        local_u = 1.0
        if i == n_show - 1:
            local_u = float(np.clip(layer_reveal_u, 0.0, 1.0))
        if local_u <= 1e-4:
            continue
        sc = float(scales[i]) * (0.35 + 0.65 * local_u)
        fa = float(face_as[i]) * local_u
        ea = float(edge_as[i]) * local_u
        ch5_draw_laplace_ellipsoid(
            ax3d,
            eigen["origin"],
            eigen["dirs"],
            base_radii * sc,
            n_u=int(CH5_ELLIPSOID_MESH_U),
            n_v=int(CH5_ELLIPSOID_MESH_V),
            face_color=face_cols[min(i, len(face_cols) - 1)],
            face_alpha=fa,
            edge_color=edge_cols[min(i, len(edge_cols) - 1)],
            edge_alpha=ea,
            zorder=12 + i,
        )
    return _fig_to_plot(fig)


def _build_laplace_ellipsoid_preview_story(clip_id):
    """ch5_64 D1 pre-rotation end → fade HPD+intervals → shells → Newton quivers → orbit."""
    frames: list = []
    hold = int(CH5_ELLIPSOID_N_HOLD)
    n_fade = int(CH5_ELLIP_FROM64_N_FADE)
    n_layer = int(CH5_ELLIPSOID_N_LAYER)
    n_quiver = int(CH5_ELLIP_FROM64_N_QUIVER)
    n_orbit = int(CH5_ELLIPSOID_N_ORBIT)
    n_shells = len(CH5_ELLIPSOID_LAYER_SCALES)
    key = "D1"
    pk = "uniform"
    mass = float(CH5_CREDIBLE_MASS)
    black = str(CH5_CRED_WANDER_PROBE_COLOR)
    smooth = _g("ch3_knob_smoothstep")
    finale_proj = _ch5_proj_states(done=("el", "st", "b"))

    pack = _PACKS[key]
    map_ws, map_we, map_bb = _ch5_map_weights(pack, prior_kind=pk)
    vox = _ch5_posterior_3d_voxel_data(
        pack["study"], pack["exam"], pack["y"], prior_kind=pk, mass=mass,
    )
    _s_lo, _s_hi, map_vec = _ch5_hpd_radial_scale_range(
        vox, map_ws, map_we, map_bb,
    )
    # ch5_64 ends on the HPD-snapped probe; Laplace geometry lives at continuous MAP.
    probe_snap = (
        float(map_vec[0]), float(map_vec[1]), float(map_vec[2]),
    )
    probe_map = (float(map_ws), float(map_we), float(map_bb))

    def _cell(
        *,
        probe,
        cloud_alpha=1.0,
        proj_states=None,
        n_layers_show=0,
        layer_reveal_u=1.0,
        show_quivers=False,
        quiver_alpha=1.0,
        cam_azim_u=0.0,
        cam_spin_deg=0.0,
    ):
        return _frame_credible_voxel(
            key,
            prior_kind=pk,
            mass=mass,
            voxel_fill_u=1.0,
            cloud_alpha=float(cloud_alpha),
            proj_states=proj_states,
            show_map=False,
            show_colormap=True,
            hq_elev=True,
            cam_azim_u=float(cam_azim_u),
            cam_spin_deg=float(cam_spin_deg),
            probe=probe,
            probe_color=black,
            dark_threshold=True,
            map_shadow_color=black,
            show_basis_quivers=bool(show_quivers),
            quiver_alpha=float(quiver_alpha),
            quiver_newton_colors=True,
            n_ellipsoid_layers=int(n_layers_show),
            ellipsoid_reveal_u=float(layer_reveal_u),
        )

    def _finish(img):
        return _ch5_finish_duo_export(img, clip_id)

    def _lerp_probe(u: float):
        t = float(np.clip(u, 0.0, 1.0))
        return tuple(
            (1.0 - t) * float(a) + t * float(b)
            for a, b in zip(probe_snap, probe_map)
        )

    # 1 — Hold ch5_64 D1 pre-rotation end (HPD + intervals + snapped probe).
    open_img = _finish(_cell(
        probe=probe_snap, cloud_alpha=1.0, proj_states=finale_proj,
    ))
    frames.extend(_hold(open_img, hold))

    # 2 — Fade region+intervals; glide probe snap → continuous MAP for Laplace.
    for tv in np.linspace(0.0, 1.0, n_fade, endpoint=True):
        u = float(smooth(float(tv)))
        a = 1.0 - u
        proj = {ax: (a, a) for ax in ("el", "st", "b")}
        frames.append(_finish(_cell(
            probe=_lerp_probe(u), cloud_alpha=a, proj_states=proj,
        )))
    clear_img = _finish(_cell(
        probe=probe_map, cloud_alpha=0.0, proj_states=None,
    ))
    frames.extend(_hold(clear_img, hold))

    # 3 — Nested Laplace shells (inner → outer), centered on continuous MAP.
    for li in range(1, n_shells + 1):
        for tv in np.linspace(0.0, 1.0, n_layer, endpoint=True):
            u = float(smooth(float(tv)))
            frames.append(_finish(_cell(
                probe=probe_map,
                cloud_alpha=0.0,
                n_layers_show=li,
                layer_reveal_u=u,
            )))
        frames.extend(_hold(frames[-1], max(1, hold // 2)))

    # 4 — Newton-colored eigen-quivers (ST blue / EL orange / b green).
    for tv in np.linspace(0.0, 1.0, n_quiver, endpoint=True):
        u = float(smooth(float(tv)))
        frames.append(_finish(_cell(
            probe=probe_map,
            cloud_alpha=0.0,
            n_layers_show=n_shells,
            layer_reveal_u=1.0,
            show_quivers=True,
            quiver_alpha=u,
        )))
    frames.extend(_hold(frames[-1], hold))

    # 5 — Orbit with shells + quivers.
    for tv in np.linspace(0.0, 1.0, n_orbit, endpoint=True):
        frames.append(_finish(_cell(
            probe=probe_map,
            cloud_alpha=0.0,
            n_layers_show=n_shells,
            layer_reveal_u=1.0,
            show_quivers=True,
            quiver_alpha=1.0,
            cam_azim_u=float(tv),
            cam_spin_deg=360.0,
        )))
    frames.extend(_hold(frames[-1], hold))
    return frames


def build_ch5_laplace_ellipsoid_preview(clip_id):
    return _build_laplace_ellipsoid_preview_story(clip_id)


def _ch5_render_d3_belief_landscape_frame():
    """One 48-style duo frame: D3 uniform belief over (w_ST, w_EL)."""
    import ch5_prior_landscape as cpl

    datasets = _ch5_landscape_grid_datasets()
    cfg = cpl.ch5_grid_landscape_config()
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    final = cpl._ch5_landscape_grid_final_pack(datasets, config=cfg)
    per_key = final["per_key"]
    pk = final["prior_kind"]
    z_color_lim = final["z_color_lim"]
    wz, ez, bz = CH5_KNOB_ZERO
    el_land = cpl._ch5_hq_land_elev()
    az_base = float(_g("CH3_LIK_W12_CT_AZIM"))
    z_lim_prior = ch5_prior_w12_z_lim(pk, scope="prior")
    empty_study = np.array([], dtype=np.float64)
    empty_exam = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)
    with cpl._landscape_render_context(cfg.dpi):
        mesh_prior = cpl.ch5_posterior_w12_mesh_pack(
            empty_study, empty_exam, empty_y, prior_kind=pk, config=cfg,
            z_lim=z_lim_prior,
        )
        z_ref_prior = float(np.nanmax(mesh_prior["Z"]))
        base_step = cpl._ch5_landscape_grid_base_step(
            ws=wz, we=ez, bb=bz,
            mesh_prior=mesh_prior,
            z_lim_prior=z_lim_prior,
            z_ref_prior=z_ref_prior,
            elev=el_land,
            azim=az_base,
            pk=pk,
            z_color_lim=z_color_lim,
        )
        return cpl._ch5_render_landscape_single(
            base_step, datasets, "D3", fk, per_key,
        )


def _ch5_d3_duo_hide_knobs(fig, axes_k):
    for ax in axes_k:
        ax.set_visible(False)
        ax.set_axis_off()
    fig.canvas.draw()


def _ch5_d3_sig_belief_cache():
    """Frozen D3 uniform belief surface + camera (content fixed; panel may resize)."""
    import ch5_prior_landscape as cpl

    datasets = _ch5_landscape_grid_datasets()
    cfg = cpl.ch5_grid_landscape_config()
    final = cpl._ch5_landscape_grid_final_pack(datasets, config=cfg)
    per_key = final["per_key"]["D3"]
    pk = final["prior_kind"]
    z_color_lim = final["z_color_lim"]
    mesh_pack = per_key["mesh_pack"]
    # Same z shelf as ch5_47+ grid finales (global peak, not prior-only).
    z_lim = ch5_belief_landscape_z_lim(pk, phase_u=1.0)
    fc, Zplot = cpl._ch5_uniform_surface_facecolors(
        mesh_pack,
        z_lim=z_lim,
        z_color_lim=z_color_lim,
        lrev=1.0,
        origin="lo_hi",
        surface_alpha=CH5_BELIEF_SURFACE_ALPHA,
    )
    gkw = cpl._ch5_uniform_surface_grid_kw(mesh_pack)
    return {
        "W1": mesh_pack["W1m"],
        "W2": mesh_pack["W2m"],
        "Z": Zplot,
        "facecolors": fc,
        "grid_kw": {
            "edgecolor": gkw["surface_grid_edgecolor"],
            "linewidth": gkw["surface_grid_linewidth"],
            "rstride": gkw["surface_grid_rstride"],
            "cstride": gkw["surface_grid_cstride"],
        },
        "z_lim": z_lim,
        "elev": float(cpl._ch5_hq_land_elev()),
        "azim": float(_g("CH3_LIK_W12_CT_AZIM")),
        "marker_ws": per_key.get("marker_ws"),
        "marker_we": per_key.get("marker_we"),
        "marker_z": per_key.get("marker_z"),
        "prior_kind": pk,
        "w1_lo": float(mesh_pack["w1_lo"]),
        "w1_hi": float(mesh_pack["w1_hi"]),
        "w2_lo": float(mesh_pack["w2_lo"]),
        "w2_hi": float(mesh_pack["w2_hi"]),
    }


def _ch5_d3_sig_draw_belief(fig, ax3d, belief):
    """Redraw the cached belief landscape on ``ax3d`` (same pose every frame)."""
    ax3d.cla()
    ax3d.computed_zorder = False
    ch5_plot_belief_surface_with_grid(
        ax3d, belief["W1"], belief["W2"], belief["Z"],
        facecolors=belief["facecolors"],
        alpha=float(CH5_BELIEF_SURFACE_ALPHA),
        zorder=5,
        **belief["grid_kw"],
    )
    _ch5_style_belief_w12_ax3d(
        ax3d, prior_kind=belief["prior_kind"], z_lim=belief["z_lim"],
    )
    ax3d.view_init(elev=float(belief["elev"]), azim=float(belief["azim"]))
    mws, mwe, mz = belief["marker_ws"], belief["marker_we"], belief["marker_z"]
    if mws is not None and mwe is not None and mz is not None:
        _ch5_draw_map_peak_marker(fig, ax3d, mws, mwe, mz, here_annotation=True)


def _ch5_d3_sig_layout_boxes(
    ax_data, ax3d, *, overlap_u=0.0, belief_shrink_u=0.0, left_dx_u=0.0,
    left_grow_u=0.0, left_is_3d=False, top_scale=None, top_dy=None,
):
    """
    2D mode: default duo boxes (never rescale the 2D plot).

    3D top-view (grow/shrink u=0): exact locked footprint
    (TOP_SCALE, TOP_DY).

    During late tilt: left grows up to LEFT_GROW (left edge fixed → expands
    right) with TILT_DX; belief shrinks uniformly toward its right edge.
    """
    d = ax_data.get_position()
    r = ax3d.get_position()
    if not left_is_3d:
        return (
            (float(d.x0), float(d.y0), float(d.width), float(d.height)),
            (float(r.x0), float(r.y0), float(r.width), float(r.height)),
        )

    bu = float(np.clip(belief_shrink_u, 0.0, 1.0))
    dxu = float(np.clip(left_dx_u, 0.0, 1.0))
    gu = float(np.clip(left_grow_u, 0.0, 1.0))
    ts = float(CH5_D3_SIG_TOP_SCALE if top_scale is None else top_scale)
    tdy = float(CH5_D3_SIG_TOP_DY if top_dy is None else top_dy)

    # --- Locked top-view footprint ---
    cx = float(d.x0) + 0.5 * float(d.width)
    cy = float(d.y0) + 0.5 * float(d.height)
    lw0 = float(d.width) * ts
    lh0 = float(d.height) * ts
    grow = 1.0 + (float(CH5_D3_SIG_LEFT_GROW) - 1.0) * gu
    lw = lw0 * grow
    lh = lh0 * grow
    # Keep the locked left edge, expand rightward; add TILT_DX with the grow.
    lx0 = cx - 0.5 * lw0
    lx = lx0 + float(CH5_D3_SIG_TILT_DX) * dxu
    ly = max(0.03, cy - 0.5 * lh + tdy)

    # Belief: uniform shrink toward its right edge (synced with left grow).
    r_s = 1.0 + (float(CH5_D3_SIG_RIGHT_SHRINK) - 1.0) * bu
    rw = float(r.width) * r_s
    rh = float(r.height) * r_s
    rx = float(r.x0 + r.width) - rw
    ry = float(r.y0) + 0.5 * (float(r.height) - rh)
    return (lx, ly, lw, lh), (rx, ry, rw, rh)


def _ch5_d3_sig_band_plan(study, exam, p_pts, *, band_step=None):
    """
    Order parallel-to-threshold bands by ascending mean σ (lowest-prob first).

    Bands are constant d = st − el. The decision threshold is inserted as its
    own band at d = 0 (target height ½).
    """
    step = float(CH5_D3_SIG_BAND_STEP if band_step is None else band_step)
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    p_pts = np.asarray(p_pts, dtype=np.float64)
    d = study - exam
    keys = np.round(d / step) * step
    uniq = sorted({float(k) for k in keys})
    if not any(abs(k) < 0.5 * step for k in uniq):
        uniq.append(0.0)
        uniq.sort()

    bands = []
    for k in uniq:
        mask = np.isclose(keys, k, atol=0.25 * step)
        mean_p = float(np.mean(p_pts[mask])) if np.any(mask) else 0.5
        is_thr = abs(float(k)) < 0.5 * step
        bands.append({
            "d": float(k),
            "mean_p": 0.5 if is_thr else mean_p,
            "idx": np.flatnonzero(mask),
            "is_threshold": is_thr,
        })
    bands.sort(key=lambda b: (float(b["mean_p"]), float(b["d"])))
    return bands


def _ch5_d3_sig_row_progress(raise_u, n_bands, *, overlap=None):
    """Per-band local progress in [0,1] for a cascading row raise."""
    u = float(np.clip(raise_u, 0.0, 1.0))
    n = max(1, int(n_bands))
    ov = float(CH5_D3_SIG_BAND_OVERLAP if overlap is None else overlap)
    ov = float(np.clip(ov, 0.15, 0.95))
    span = float(n - 1) + ov
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        local = (u * span - float(i)) / ov
        out[i] = float(np.clip(local, 0.0, 1.0))
    return out


def _ch5_d3_sig_heights(raise_u, p_pts, bands, *, smooth=None):
    """Point heights + threshold height for the current raise progress."""
    sm = _g("ch3_knob_smoothstep") if smooth is None else smooth
    p_pts = np.asarray(p_pts, dtype=np.float64)
    z_pts = np.zeros_like(p_pts)
    thr_z = 0.0
    prog = _ch5_d3_sig_row_progress(raise_u, len(bands))
    for i, band in enumerate(bands):
        lu = float(sm(float(prog[i])))
        if band["is_threshold"]:
            thr_z = lu * 0.5
        idx = band["idx"]
        if idx.size:
            z_pts[idx] = lu * p_pts[idx]
    return z_pts, float(thr_z)


def _ch5_frame_d3_sig_duo(
    *,
    overlap_u=0.0,
    belief_shrink_u=0.0,
    left_dx_u=0.0,
    left_grow_u=0.0,
    raise_u=0.0,
    surf_u=0.0,
    elev=26.0,
    azim=25.0,
    mesh=None,
    study=None,
    exam=None,
    y=None,
    xlim=None,
    ylim=None,
    check_icon=None,
    cross_icon=None,
    p_pts=None,
    bands=None,
    ws=None,
    we=None,
    bb=None,
    belief=None,
    left_mode="2d",
    show_knobs=True,
    top_scale=None,
    top_dy=None,
):
    """
    Duo frame: left data/sigmoid + right belief landscape.

    The 2D plot never rescales. Left 3D top-view uses a scaled/shifted copy of
    the 2D data footprint; on tilt ``left_grow_u`` grows it (and ``left_dx_u``
    shifts it right) while ``belief_shrink_u`` shrinks the right belief panel.
    """
    from hita.primitives.colormap import CMAP
    from hita.primitives.sigmoid_3d import (
        draw_sigmoid_surface,
        scatter_outcome_icons_3d,
        style_sigmoid_axes,
    )

    fig, ax_data, ax3d, axes_k = _g("ch4_figure_duo_weight3d")()
    mode = str(left_mode).lower()
    left_is_3d = mode == "3d"
    ou = float(np.clip(overlap_u, 0.0, 1.0)) if left_is_3d else 0.0
    bu = float(np.clip(belief_shrink_u, 0.0, 1.0)) if left_is_3d else 0.0
    dxu = float(np.clip(left_dx_u, 0.0, 1.0)) if left_is_3d else 0.0
    gu = float(np.clip(left_grow_u, 0.0, 1.0)) if left_is_3d else 0.0
    left_box, right_box = _ch5_d3_sig_layout_boxes(
        ax_data, ax3d, overlap_u=ou, belief_shrink_u=bu, left_dx_u=dxu,
        left_grow_u=gu, left_is_3d=left_is_3d,
        top_scale=top_scale, top_dy=top_dy,
    )
    want_knobs = bool(show_knobs) and (not left_is_3d)

    if want_knobs:
        ax_data.set_position(left_box)
        ax3d.set_position(right_box)
    else:
        _ch5_d3_duo_hide_knobs(fig, axes_k)
        ax_data.set_position(left_box)
        ax3d.set_position(right_box)

    # Belief first (may sit under an overlapping left 3D panel).
    ax3d.set_zorder(1)
    _ch5_d3_sig_draw_belief(fig, ax3d, belief)

    if not left_is_3d:
        _ch5_draw_data_panel(
            ax_data, study, exam, y,
            xl=xlim, yl=ylim,
            show_colormap=False,
            show_threshold=True,
            ws=float(ws), we=float(we), bb=float(bb),
        )
        ax_data.set_xlim(float(xlim[0]), float(xlim[1]))
        ax_data.set_ylim(float(ylim[0]), float(ylim[1]))
        try:
            ax_data.set_aspect("equal", adjustable="box")
        except Exception:
            pass
        if want_knobs:
            knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
            _g("ch3_draw_knob_row")(
                fig, axes_k, float(ws), float(we), float(bb), "all",
                knob_rgbs, canvas_sides,
                rot_strip_deg=0.0, strip_scale=1.0,
                knob_rots=_g("ch3_k1_knob_rots_at")(float(ws), float(we), float(bb)),
                knob_scales=[1.0, 1.0, 1.0], ax_data=ax_data,
            )
        return _fig_to_plot(fig)

    # Left 3D in the data-panel footprint (then overlapping on tilt).
    ax_data.remove()
    axL = fig.add_axes(left_box, projection="3d")
    # Opaque while matching top-view; fade panel fill as we overlap belief.
    panel_a = float(1.0 - 0.85 * ou)
    axL.set_facecolor((1.0, 1.0, 1.0, panel_a))
    try:
        axL.patch.set_alpha(panel_a)
    except Exception:
        pass
    axL.computed_zorder = False
    axL.set_zorder(3)

    ru = float(np.clip(raise_u, 0.0, 1.0))
    su = float(np.clip(surf_u, 0.0, 1.0))
    elev_f = float(elev)
    topdown = elev_f >= 86.0
    draw_surf = su > 1e-4
    # Locked zoom at top-view; ease toward a slightly pulled-back orbit zoom so
    # cube corners / σ edges stay inside the axes rectangle at all azims.
    z0 = float(CH5_D3_SIG_LEFT_ZOOM)
    z1 = float(CH5_D3_SIG_LEFT_ZOOM_ORBIT)
    # Ease with tilt (same schedule as aspect); grow_u alone stays 0 early in tilt.
    elev_lo = float(CH5_D3_SIG_ELEV_3D)
    elev_hi = float(CH5_D3_SIG_ELEV_FLAT)
    if elev_hi > elev_lo + 1e-9:
        t_tilt = float(np.clip((elev_hi - elev_f) / (elev_hi - elev_lo), 0.0, 1.0))
    else:
        t_tilt = 0.0 if topdown else 1.0
    zoom = z0 + (z1 - z0) * t_tilt

    if draw_surf:
        Z = mesh.morph_z(su, pass_surface=True)
        alpha = 0.18 + 0.22 * su
        surf = draw_sigmoid_surface(axL, mesh, Z, cmap=CMAP, alpha=float(alpha))
        try:
            surf.set_clip_on(False)
        except Exception:
            pass
    else:
        Z0 = np.zeros_like(mesh.ST)
        axL.plot_surface(
            mesh.ST, mesh.EL, Z0,
            alpha=0.0, linewidth=0, antialiased=False, shade=False,
        )

    if bands is None:
        bands = _ch5_d3_sig_band_plan(study, exam, p_pts)
    z_pts, thr_z = _ch5_d3_sig_heights(ru, p_pts, bands)
    span0 = float(CH5_D3_SIG_ICON_SPAN_FRAC)
    span1 = float(CH5_D3_SIG_ICON_SPAN_FRAC_POST)
    span_frac = span0 + (span1 - span0) * gu
    scatter_outcome_icons_3d(
        axL, study, exam, y, z_pts,
        check_icon=check_icon,
        cross_icon=cross_icon,
        xlim=xlim,
        ylim=ylim,
        rotate_icons_180=True,
        span_frac=float(span_frac),
    )
    # Icons are Poly3DCollections; keep them from getting snipped with the σ sheet.
    try:
        for coll in list(axL.collections):
            coll.set_clip_on(False)
    except Exception:
        pass

    if ws is not None and we is not None and abs(float(we)) > 1e-9:
        x0, x1 = float(xlim[0]), float(xlim[1])
        y0 = -(float(ws) * x0 + float(bb)) / float(we)
        y1 = -(float(ws) * x1 + float(bb)) / float(we)
        thr_line = axL.plot(
            [x0, x1], [y0, y1], [thr_z, thr_z],
            color="#222222", linestyle="--", linewidth=1.6, zorder=20,
        )
        try:
            for ln in thr_line:
                ln.set_clip_on(False)
        except Exception:
            pass

    zlo, zhi = (0.0, 1.0) if topdown else tuple(float(v) for v in CH5_D3_SIG_ZLIM)

    style_sigmoid_axes(
        axL, float(azim),
        elev=elev_f,
        xlim=xlim,
        ylim=ylim,
        hide_z=topdown or elev_f >= 72.0,
        exam_label_2d=False,
        # Own threshold above tracks thr_z; skip the fixed floor diagonal so
        # nothing stays at z=0 after the line lifts.
        show_threshold=False,
        font_size=float(_g("FONT_SIZE")) * 0.92,
        axis_label_size=float(_g("AXIS_LABEL_SIZE")) * 0.92,
        diag_prob_scale=0.0,
    )
    if topdown:
        # Match 2D: Exam length ticks/label on the left (default 3D puts them right).
        try:
            axL.yaxis.set_ticks_position("upper")
            axL.yaxis.set_label_position("upper")
        except Exception:
            pass
        try:
            axL.set_proj_type("ortho")
        except Exception:
            pass
    try:
        # Top-down: nearly flat to match 2D. Ease toward equal visual axis
        # lengths (z same as x/y) as we tilt — data ranges unchanged.
        z_asp = 0.02 + (0.75 - 0.02) * t_tilt  # post-tilt z = ¾ of x/y length
        axL.set_box_aspect(
            (1.0, 1.0, float(z_asp)),
            zoom=float(zoom),
        )
    except Exception:
        try:
            axL.set_box_aspect((1.0, 1.0, 0.02 if topdown else 0.75))
        except Exception:
            pass
    axL.set_xlim(float(xlim[0]), float(xlim[1]))
    axL.set_ylim(float(ylim[0]), float(ylim[1]))
    axL.set_zlim(float(zlo), float(zhi))
    # Avoid subplot-rectangle snipping of cube corners at some azimuths.
    try:
        axL.set_clip_on(False)
        axL.patch.set_clip_on(False)
    except Exception:
        pass
    return _fig_to_plot(fig)


def _build_d3_belief_to_sigmoid_orbit_story(clip_id):
    """
    2D plot stays fixed size. Left becomes a top-down 3D view sized to match
    that 2D plot, then tilts/raises/orbits in place (no panel slide). Parallel
    bands raise low-σ first; threshold → ½; colormap after points are up.
    """
    from hita.primitives.icons import load_icon_arrays
    from hita.primitives.math import sigmoid
    from hita.primitives.sigmoid_3d import SigmoidMesh

    frames: list = []
    hold = int(CH5_D3_SIG_N_HOLD)
    n_cross = int(CH5_D3_SIG_N_CROSS)
    n_tilt = int(CH5_D3_SIG_N_TILT)
    n_raise = int(CH5_D3_SIG_N_RAISE)
    n_surf = int(CH5_D3_SIG_N_SURF)
    n_orbit = int(CH5_D3_SIG_N_ORBIT)
    elev_flat = float(CH5_D3_SIG_ELEV_FLAT)
    elev_3d = float(CH5_D3_SIG_ELEV_3D)
    # One-shot tilt nudge before the spin; elev stays fixed during orbit (Ch1-style).
    elev_orbit = elev_3d - float(CH5_D3_SIG_ORBIT_ELEV_DELTA)
    az_top = float(CH5_D3_SIG_AZ_TOP)
    az0 = float(CH5_D3_SIG_AZ0)
    # Shortest signed azimuth delta top-down → post-tilt init (degrees, (−180, 180]).
    daz = ((az0 - az_top + 180.0) % 360.0) - 180.0
    smooth = _g("ch3_knob_smoothstep")

    _ensure_packs()
    pack = _PACKS["D3"]
    study = np.asarray(pack["study"], dtype=np.float64)
    exam = np.asarray(pack["exam"], dtype=np.float64)
    y = np.asarray(pack["y"], dtype=np.int64)
    xlim, ylim = ch5_plot_limits("D3")
    ws, we, bb = _ch5_map_weights(pack, prior_kind="uniform")
    # Mesh = axis limits (Ch1 DNA). Padding past lims overhangs/clips at corners.
    pad = float(CH5_D3_SIG_MESH_PAD)
    mesh = SigmoidMesh.build(
        (float(xlim[0]) - pad, float(xlim[1]) + pad),
        (float(ylim[0]) - pad, float(ylim[1]) + pad),
        n=int(CH5_D3_SIG_MESH_N),
        w_st=float(ws), w_el=float(we), b=float(bb),
    )
    logits = float(ws) * study + float(we) * exam + float(bb)
    p_pts = sigmoid(logits)
    bands = _ch5_d3_sig_band_plan(study, exam, p_pts)
    check_icon, cross_icon = load_icon_arrays()
    belief = _ch5_d3_sig_belief_cache()

    def _frame(
        *,
        overlap_u=0.0,
        belief_shrink_u=0.0,
        left_dx_u=0.0,
        left_grow_u=0.0,
        raise_u=0.0,
        surf_u=0.0,
        elev=elev_3d,
        azim=az0,
        left_mode="2d",
        show_knobs=True,
    ):
        return _ch5_finish_duo_export(
            _ch5_frame_d3_sig_duo(
                overlap_u=overlap_u,
                belief_shrink_u=belief_shrink_u,
                left_dx_u=left_dx_u,
                left_grow_u=left_grow_u,
                raise_u=raise_u, surf_u=surf_u, elev=elev, azim=azim,
                mesh=mesh, study=study, exam=exam, y=y,
                xlim=xlim, ylim=ylim,
                check_icon=check_icon, cross_icon=cross_icon,
                p_pts=p_pts, bands=bands,
                ws=ws, we=we, bb=bb,
                belief=belief,
                left_mode=left_mode,
                show_knobs=show_knobs,
            ),
            clip_id,
        )

    # --- Opening: 2D at default size (never rescaled) + belief ---
    open_fr = _frame(left_mode="2d", show_knobs=True)
    frames.extend(_hold(open_fr, hold))

    # --- 2D → top-down 3D (same footprint; zoom matches 2D size) ---
    # Hide knobs on the 2D side of the crossfade so only the plot morphs.
    flat_2d = _frame(left_mode="2d", show_knobs=False)
    top = _frame(
        overlap_u=0.0, belief_shrink_u=0.0, left_dx_u=0.0, left_grow_u=0.0,
        raise_u=0.0, surf_u=0.0,
        elev=elev_flat, azim=az_top, left_mode="3d", show_knobs=False,
    )
    for tv in np.linspace(0.0, 1.0, n_cross, endpoint=True):
        u = float(smooth(float(tv)))
        frames.append(ch5_crossfade_images(flat_2d, top, u))
    frames.extend(_hold(top, hold))

    # --- Tilt; late in the tilt, grow left 20% + shift right, shrink belief ---
    g0 = float(CH5_D3_SIG_GROW_START)
    for tv in np.linspace(0.0, 1.0, n_tilt, endpoint=True):
        u = float(smooth(float(tv)))
        elev = (1.0 - u) * elev_flat + u * elev_3d
        az = az_top + daz * u
        # Size morph mostly in the latter part of the tilt.
        t_grow = float(np.clip((u - g0) / max(1.0 - g0, 1e-9), 0.0, 1.0))
        grow_u = float(smooth(t_grow))
        frames.append(_frame(
            overlap_u=0.0,
            belief_shrink_u=grow_u,
            left_dx_u=grow_u,
            left_grow_u=grow_u,
            raise_u=0.0, surf_u=0.0,
            elev=elev, azim=az, left_mode="3d", show_knobs=False,
        ))
    frames.extend(_hold(frames[-1], max(2, hold // 2)))

    # --- Raise bands (low-σ first); threshold lifts to ½ ---
    for tv in np.linspace(0.0, 1.0, n_raise, endpoint=True):
        u = float(smooth(float(tv)))
        frames.append(_frame(
            overlap_u=0.0, belief_shrink_u=1.0, left_dx_u=1.0, left_grow_u=1.0,
            raise_u=u, surf_u=0.0,
            elev=elev_3d, azim=az0, left_mode="3d", show_knobs=False,
        ))
    frames.extend(_hold(frames[-1], max(2, hold // 2)))

    # --- Colormap after all points have height ---
    for tv in np.linspace(0.0, 1.0, n_surf, endpoint=True):
        u = float(smooth(float(tv)))
        frames.append(_frame(
            overlap_u=0.0, belief_shrink_u=1.0, left_dx_u=1.0, left_grow_u=1.0,
            raise_u=1.0, surf_u=u,
            elev=elev_3d, azim=az0, left_mode="3d", show_knobs=False,
        ))
    frames.extend(_hold(frames[-1], hold))

    # --- Orbit left: full 360° CCW at fixed elev (δ applied once, no mid-spin tilt) ---
    for tv in np.linspace(0.0, 1.0, n_orbit, endpoint=True):
        az = az0 + 360.0 * float(tv)
        frames.append(_frame(
            overlap_u=0.0, belief_shrink_u=1.0, left_dx_u=1.0, left_grow_u=1.0,
            raise_u=1.0, surf_u=1.0,
            elev=elev_orbit, azim=az, left_mode="3d", show_knobs=False,
        ))
    frames.extend(_hold(frames[-1], hold))
    return frames



def build_ch5_d3_belief_to_sigmoid_orbit(clip_id):
    return _build_d3_belief_to_sigmoid_orbit_story(clip_id)


def _build_d1_fanout_credible_story(clip_id):
    """
    D1 + HPD credible region → glide all students into the E2 triangle fills
    (pass: ST∈[0,1] EL∈[0,3]; fail: ST∈[0,1] EL∈[3,7]), then batch-add the
    remaining E2 points. Credible voxels/MAP recompute every frame during the
    glide and after each add batch. Opens with a 60° CCW camera turn (held
    through edits), then a full 360° orbit.
    """
    frames: list = []
    hold = int(CH5_D1_FAN_N_HOLD)
    n_move = int(CH5_D1_FAN_N_MOVE)
    n_add_morph = int(CH5_D1_FAN_N_ADD_MORPH)
    add_hold = int(CH5_D1_FAN_N_ADD_HOLD)
    add_batch = max(1, int(CH5_D1_FAN_ADD_BATCH))
    n_setup = int(CH5_D1_FAN_N_SETUP_ROT)
    n_orbit = int(CH5_D1_FAN_N_ORBIT)
    setup_spin = float(CH5_D1_FAN_SETUP_SPIN)
    orbit_deg = float(CH5_D1_FAN_ORBIT_DEG)
    pk = "uniform"
    mass = float(CH5_CREDIBLE_MASS)
    smooth = _g("ch3_knob_smoothstep")
    cu0, cs0 = _ch5_cam_from_spin(setup_spin)

    _ensure_packs()
    pack = _PACKS["D1"]
    study0 = np.asarray(pack["study"], dtype=np.float64).copy()
    exam0 = np.asarray(pack["exam"], dtype=np.float64).copy()
    y = np.asarray(pack["y"], dtype=np.int64).copy()
    move_to = np.asarray(CH5_D1_FAN_MOVE_TO, dtype=np.float64)
    if move_to.shape[0] != study0.shape[0]:
        raise ValueError(
            f"CH5_D1_FAN_MOVE_TO length {move_to.shape[0]} != D1 n={study0.shape[0]}"
        )
    study1 = move_to[:, 0].copy()
    exam1 = move_to[:, 1].copy()
    adds = list(CH5_D1_FAN_ADDS)

    def _cell(st, el, yy, *, cam_azim_u=0.0, cam_spin_deg=0.0):
        return _frame_credible_voxel(
            "D1",
            study=st, exam=el, y=yy,
            prior_kind=pk,
            mass=mass,
            voxel_fill_u=1.0,
            proj_states=None,
            show_map=True,
            show_colormap=True,
            hq_elev=True,
            cam_azim_u=float(cam_azim_u),
            cam_spin_deg=float(cam_spin_deg),
        )

    # Opening at og view, then setup spin; hold that pose through the edits.
    open0 = _ch5_finish_duo_export(
        _cell(study0, exam0, y, cam_azim_u=0.0, cam_spin_deg=0.0), clip_id,
    )
    frames.extend(_hold(open0, hold))
    for tv in np.linspace(0.0, 1.0, n_setup, endpoint=True):
        u = float(smooth(float(tv)))
        spin = setup_spin * u
        cu, cs = _ch5_cam_from_spin(spin)
        frames.append(_ch5_finish_duo_export(
            _cell(study0, exam0, y, cam_azim_u=cu, cam_spin_deg=cs), clip_id,
        ))
    cur = _ch5_finish_duo_export(
        _cell(study0, exam0, y, cam_azim_u=cu0, cam_spin_deg=cs0), clip_id,
    )
    frames.extend(_hold(cur, hold))

    # Parallel glide: every D1 point moves into its E2 triangle slot; HPD tracks.
    for tv in np.linspace(0.0, 1.0, n_move, endpoint=True):
        u = float(smooth(float(tv)))
        study = study0 + u * (study1 - study0)
        exam = exam0 + u * (exam1 - exam0)
        cur = _ch5_finish_duo_export(
            _cell(study, exam, y, cam_azim_u=cu0, cam_spin_deg=cs0), clip_id,
        )
        frames.append(cur)
    study = study1.copy()
    exam = exam1.copy()
    frames.extend(_hold(cur, hold))

    # Batch-add remaining E2 points (quick morphs).
    for i0 in range(0, len(adds), add_batch):
        batch = adds[i0 : i0 + add_batch]
        for st_a, el_a, lab_a in batch:
            study = np.append(study, float(st_a))
            exam = np.append(exam, float(el_a))
            y = np.append(y, int(lab_a)).astype(np.int64)
        nxt = _ch5_finish_duo_export(
            _cell(study, exam, y, cam_azim_u=cu0, cam_spin_deg=cs0), clip_id,
        )
        for tv in np.linspace(0.0, 1.0, n_add_morph, endpoint=True):
            u = float(smooth(float(tv)))
            frames.append(ch5_crossfade_images(cur, nxt, u))
        frames.extend(_hold(nxt, add_hold))
        cur = nxt

    frames.extend(_hold(cur, hold))

    # Full 360° CCW orbit continuing from the setup pose.
    for tv in np.linspace(0.0, 1.0, n_orbit, endpoint=True):
        spin = setup_spin + orbit_deg * float(tv)
        cu, cs = _ch5_cam_from_spin(spin)
        frames.append(_ch5_finish_duo_export(
            _cell(study, exam, y, cam_azim_u=cu, cam_spin_deg=cs),
            clip_id,
        ))
    frames.extend(_hold(frames[-1], hold))
    return frames


def build_ch5_d1_fanout_credible(clip_id):
    return _build_d1_fanout_credible_story(clip_id)


def _ch5_d2_noise_indices(study, exam, y, targets):
    """Map (st, el, label) tuples to indices in the D2 roster."""
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    out = []
    for st_t, el_t, lab_t in targets:
        d2 = (study - float(st_t)) ** 2 + (exam - float(el_t)) ** 2
        mask = y == int(lab_t)
        if not np.any(mask):
            raise ValueError(f"D2 noise point missing: {(st_t, el_t, lab_t)}")
        j_local = int(np.argmin(d2[mask]))
        j = int(np.flatnonzero(mask)[j_local])
        out.append(j)
    return out


def _ch5_with_outcome_icon_scales(scales, render_fn):
    """Temporarily draw 2D check/cross icons with per-point zoom multipliers."""
    base_draw = _g("draw_dataset")
    add_icon = _g("add_outcome_icon")
    sc = None if scales is None else np.asarray(scales, dtype=np.float64)

    def _draw(ax, study_vals, exam_vals, labels, mask=None, alpha=0.95, icon_zoom=0.2):
        study_vals = np.asarray(study_vals, dtype=np.float64)
        exam_vals = np.asarray(exam_vals, dtype=np.float64)
        labels = np.asarray(labels)
        n = int(labels.shape[0])
        m = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        base_z = float(icon_zoom)
        for i in range(n):
            if not bool(m[i]):
                continue
            mul = 1.0 if sc is None else float(sc[i])
            add_icon(
                ax,
                float(study_vals[i]),
                float(exam_vals[i]),
                bool(int(labels[i]) == 1),
                zoom=base_z * mul,
                alpha=float(alpha),
            )

    _G["draw_dataset"] = _draw
    try:
        return render_fn()
    finally:
        _G["draw_dataset"] = base_draw


def _build_d2_noise_icon_emphasize_story(clip_id):
    """
    Zoomed D2 belief duo (same pose as ch5_47/48 D2 zoom hold): grow the two
    pass icons on the fail side of the MAP threshold, shrink back, then grow the
    two fail icons on the pass side, shrink back.
    """
    frames: list = []
    hold = int(CH5_D2_NOISE_EMPH_N_HOLD)
    n_grow = int(CH5_D2_NOISE_EMPH_N_GROW)
    peak = float(CH5_D2_NOISE_EMPH_SCALE)
    smooth = _g("ch3_knob_smoothstep")
    cfg = ch5_prior_landscape.ch5_grid_landscape_config()
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    datasets = _ch5_landscape_grid_datasets()
    final = ch5_prior_landscape._ch5_landscape_grid_final_pack(datasets, config=cfg)
    per_key = final["per_key"]
    pk = final["prior_kind"]
    z_color_lim = final["z_color_lim"]
    wz, ez, bz = CH5_KNOB_ZERO
    el_land = ch5_prior_landscape._ch5_hq_land_elev()
    az_base = float(_g("CH3_LIK_W12_CT_AZIM"))
    z_lim_prior = ch5_prior_landscape.ch5_prior_w12_z_lim(pk, scope="prior")

    ds = datasets["D2"]
    study = np.asarray(ds["study"], dtype=np.float64)
    exam = np.asarray(ds["exam"], dtype=np.float64)
    y = np.asarray(ds["y"], dtype=np.int64)
    pass_idx = _ch5_d2_noise_indices(study, exam, y, CH5_D2_NOISE_PASS)
    fail_idx = _ch5_d2_noise_indices(study, exam, y, CH5_D2_NOISE_FAIL)
    n_pts = int(y.shape[0])

    empty_study = np.array([], dtype=np.float64)
    empty_exam = np.array([], dtype=np.float64)
    empty_y = np.array([], dtype=np.int64)

    with ch5_prior_landscape._landscape_render_context(cfg.dpi):
        mesh_prior = ch5_prior_landscape.ch5_posterior_w12_mesh_pack(
            empty_study, empty_exam, empty_y, prior_kind=pk, config=cfg,
            z_lim=z_lim_prior,
        )
        z_ref_prior = float(np.nanmax(mesh_prior["Z"]))
        base_step = ch5_prior_landscape._ch5_landscape_grid_base_step(
            ws=wz, we=ez, bb=bz,
            mesh_prior=mesh_prior,
            z_lim_prior=z_lim_prior,
            z_ref_prior=z_ref_prior,
            elev=el_land,
            azim=az_base,
            pk=pk,
            z_color_lim=z_color_lim,
        )

        def _render(scales):
            def _once():
                return ch5_prior_landscape._ch5_render_landscape_single(
                    base_step, datasets, "D2", fk, per_key,
                )
            img = _ch5_with_outcome_icon_scales(scales, _once)
            return _ch5_finish_duo_export(img, clip_id)

        def _scales_for(idxs, mul):
            sc = np.ones(n_pts, dtype=np.float64)
            for j in idxs:
                sc[int(j)] = float(mul)
            return sc

        def _pulse(idxs):
            for tv in np.linspace(0.0, 1.0, n_grow, endpoint=True):
                u = float(smooth(float(tv)))
                mul = 1.0 + (peak - 1.0) * u
                frames.append(_render(_scales_for(idxs, mul)))
            peak_sc = _scales_for(idxs, peak)
            frames.extend(_hold(_render(peak_sc), hold))
            for tv in np.linspace(0.0, 1.0, n_grow, endpoint=True):
                u = float(smooth(float(tv)))
                mul = peak + (1.0 - peak) * u
                frames.append(_render(_scales_for(idxs, mul)))
            frames.extend(_hold(_render(None), max(1, hold // 2)))

        frames.extend(_hold(_render(None), hold))
        _pulse(pass_idx)  # checks left of threshold
        _pulse(fail_idx)  # crosses right of threshold
        frames.extend(_hold(_render(None), hold))

    return frames


def build_ch5_d2_noise_icon_emphasize(clip_id):
    return _build_d2_noise_icon_emphasize_story(clip_id)


def _ch5_line_through_point(angle_rad, cx, cy, *, scale=1.0):
    """Decision boundary through (cx, cy) with normal angle ``angle_rad``."""
    ws = float(scale) * float(np.cos(angle_rad))
    we = float(scale) * float(np.sin(angle_rad))
    bb = -(ws * float(cx) + we * float(cy))
    return ws, we, bb


def _ch5_d4_rotating_cuts(n_cuts, *, scale=1.0):
    """Evenly spaced orientations through D4's circle center."""
    cx, cy = CH5_D4_CENTER
    n = max(1, int(n_cuts))
    out = []
    for i in range(n):
        ang = (np.pi * i) / n  # half-turn covers unique lines (normals π-apart coincide)
        out.append(_ch5_line_through_point(ang, cx, cy, scale=scale))
    return out


def _ch5_dashed_threshold_extra(ws, we, bb, *, linewidth=None, alpha=0.82):
    """Grey dashed threshold tuple for ``extra_thresholds``."""
    lw = float(CH5_BEST_LINE_LW_THIN if linewidth is None else linewidth)
    return (float(ws), float(we), float(bb), "grey", "--", lw, float(alpha))


def _frame_d1_wide_to_duo_empty3d(
    layout_u,
    *,
    ws,
    we,
    bb,
    threshold_linewidth=None,
    threshold_legend=False,
    show_3d_u=0.0,
):
    """
    Morph wide full-bleed D1+threshold → ch4 weight3d duo (empty Belief axes).
    ``layout_u`` 0 = wide 2D only; 1 = duo left 2D + knobs (right slot empty).
    ``show_3d_u`` fades the empty 3D Belief axes in after the resize (0→1).
    """
    u = float(np.clip(float(layout_u), 0.0, 1.0))
    su = float(_g("ch3_knob_smoothstep")(u))
    s3 = float(np.clip(float(show_3d_u), 0.0, 1.0))
    s3 = float(_g("ch3_knob_smoothstep")(s3))
    wide_data, wide_knobs = _g("_ch4_02b_wide_layout")()
    duo_data, duo_knobs, duo_3d = _ch5_weight3d_duo_layout()
    data_r = _g("_ch4_02b_lerp_rect")(su, wide_data, duo_data)
    knob_rs = tuple(_g("_ch4_02b_lerp_rect")(su, wide_knobs[i], duo_knobs[i]) for i in range(3))

    pack = _PACKS["D1"]
    study = np.asarray(pack["study"], dtype=np.float64)
    exam = np.asarray(pack["exam"], dtype=np.float64)
    y = np.asarray(pack["y"], dtype=np.int64)
    xl, yl = ch5_plot_limits("D1")
    lw = float(CH5_BEST_LINE_LW_NORMAL if threshold_linewidth is None else threshold_linewidth)

    fig = plt.figure(figsize=CH4_DUO_FIGSIZE)
    fig.patch.set_facecolor("white")
    ax_data = fig.add_axes(data_r)
    _ch5_draw_data_panel(
        ax_data, study, exam, y, xl=xl, yl=yl,
        ws=float(ws), we=float(we), bb=float(bb),
        show_threshold=True, threshold_legend=bool(threshold_legend),
        threshold_linewidth=lw, threshold_linestyle="--", threshold_color="grey",
    )
    if su > 1e-4:
        _ch5_place_knobs_flyin(
            fig, ax_data, data_r, knob_rs, float(ws), float(we), float(bb), grow_u=su,
        )
    if s3 > 1e-3:
        ax3d = fig.add_axes(duo_3d, projection="3d")
        _ch5_style_belief_w12_ax3d(
            ax3d, prior_kind="uniform",
            z_lim=ch5_prior_w12_z_lim("uniform", scope="prior"),
        )
        el = float(ch5_prior_landscape._ch5_hq_land_elev())
        az = float(_g("CH3_LIK_W12_CT_AZIM"))
        ax3d.view_init(elev=el, azim=az)
        ax3d.patch.set_alpha(s3)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.pane.fill = False
            axis.pane.set_alpha(s3 * 0.35)
    return _fig_to_plot(fig)


def _build_d1_best_line_howithink_d4_cuts_story(clip_id):
    """
    D1 → MAP → \"best line\" confetti → grey + howithinkabout → D4 rotating dashed
    cuts → fade → D1 parallel dashed family (one-at-a-time, no ±bias edges) →
    normal MAP → wide→duo resize → empty 3D appear → D1 landscape wipe (like ch5_55).
    """
    frames: list = []
    hold = int(CH5_BEST_LINE_N_HOLD)
    n_thresh = int(CH5_BEST_LINE_N_THRESH)
    n_celeb = int(CH5_BEST_LINE_N_CELEBRATE)
    n_grey = int(CH5_BEST_LINE_N_GREY_LOGO)
    n_fade = int(CH5_BEST_LINE_N_FADE)
    n_cut_hold = int(CH5_BEST_LINE_N_D4_CUT)
    n_cuts = int(CH5_BEST_LINE_N_D4_CUTS)
    n_par = int(CH5_BEST_LINE_N_PARALLEL)
    n_layout = int(CH5_BEST_LINE_N_LAYOUT)
    n_land = int(CH5_BEST_LINE_N_LAND_REVEAL)
    bias = float(CH5_BEST_LINE_PARALLEL_BIAS)
    logo_frac = float(CH5_BEST_LINE_LOGO_FRAC)
    logo_cx = float(CH5_BEST_LINE_LOGO_CX)
    lw_thick = float(CH5_BEST_LINE_LW_THICK)
    lw_norm = float(CH5_BEST_LINE_LW_NORMAL)
    lw_thin = float(CH5_BEST_LINE_LW_THIN)
    smooth = _g("ch3_knob_smoothstep")

    pack1 = _PACKS["D1"]
    n1 = len(pack1["order"])
    map1_ws, map1_we, map1_bb = _ch5_map_weights(pack1)
    study1 = np.asarray(pack1["study"], dtype=np.float64)
    exam1 = np.asarray(pack1["exam"], dtype=np.float64)
    y1 = np.asarray(pack1["y"], dtype=np.int64)

    pack4 = _PACKS["D4"]
    n4 = len(pack4["order"])
    # D4 continuous MAP is ~origin; keep a visible "best" line as D1's MAP
    # orientation through the D4 circle center.
    cx4, cy4 = CH5_D4_CENTER
    cut_scale = max(float(np.hypot(map1_ws, map1_we)), 0.85)
    map4_ws = cut_scale * float(map1_ws) / max(float(np.hypot(map1_ws, map1_we)), 1e-9)
    map4_we = cut_scale * float(map1_we) / max(float(np.hypot(map1_ws, map1_we)), 1e-9)
    map4_bb = -(map4_ws * float(cx4) + map4_we * float(cy4))
    rotating = _ch5_d4_rotating_cuts(n_cuts, scale=cut_scale)
    best_ang = float(np.arctan2(map4_we, map4_ws))
    filtered = []
    for cws, cwe, cbb in rotating:
        ang = float(np.arctan2(cwe, cws))
        line_diff = min(
            abs((ang - best_ang) % np.pi),
            np.pi - abs((ang - best_ang) % np.pi),
        )
        if line_diff < (np.pi / max(n_cuts, 1)) * 0.45:
            continue
        filtered.append((cws, cwe, cbb))
    rotating = filtered if filtered else rotating

    # D1 parallels: interior offsets only (drop ±bias edges); MAP is separate.
    # Same one-at-a-time reveal cadence as D4 rotating cuts.
    par_extras = []
    for off in np.linspace(-bias, bias, n_cuts, endpoint=True):
        if abs(float(off)) < 1e-9:
            continue
        if abs(abs(float(off)) - float(bias)) < 1e-9:
            continue  # omit the two edge-of-range thresholds
        par_extras.append(
            _ch5_dashed_threshold_extra(map1_ws, map1_we, map1_bb + float(off), linewidth=lw_thin)
        )

    def emit(img):
        return _ch5_finish_duo_export(img, clip_id)

    def add(img, *, n=None):
        frames.append(emit(img))
        if n is not None and int(n) > 1:
            frames.extend(_hold(frames[-1], int(n) - 1))

    def plot_d1(*, show_threshold=False, extra=None, legend=True, thr_lw=None):
        return _frame_plot_2d_dataset(
            "D1", n_show=n1, ws=map1_ws, we=map1_we, bb=map1_bb,
            show_threshold=show_threshold, extra_thresholds=extra,
            threshold_legend=legend,
            threshold_linewidth=lw_norm if thr_lw is None else thr_lw,
            threshold_linestyle="--", threshold_color="grey",
        )

    def plot_d4(*, show_threshold=False, extra=None, thr_lw=None):
        return _frame_plot_2d_dataset(
            "D4", n_show=n4, ws=map4_ws, we=map4_we, bb=map4_bb,
            show_threshold=show_threshold, extra_thresholds=extra,
            threshold_legend=False,
            threshold_linewidth=lw_thick if thr_lw is None else thr_lw,
            threshold_linestyle="--", threshold_color="grey",
        )

    # --- D1 open + MAP threshold ---
    d1_bare = plot_d1(show_threshold=False)
    add(d1_bare, n=hold)
    d1_map = plot_d1(show_threshold=True, legend=True, thr_lw=lw_norm)
    add(d1_map, n=n_thresh)

    # --- Celebrate: "best line" + confetti ---
    for i in range(max(1, n_celeb)):
        u = smooth(float(i) / max(n_celeb - 1, 1))
        frames.append(emit(ch5_confetti_best_line_overlay(d1_map, u=u, label="best line")))
    frames.extend(_hold(frames[-1], hold))

    # --- Grey out + howithinkabout logo (clean base, no confetti) ---
    for i in range(max(1, n_grey)):
        u = smooth(float(i + 1) / max(n_grey, 1))
        frames.append(emit(ch5_overlay_howithink_center_right(
            d1_map, dim_u=u, logo_u=u, size_frac=logo_frac, cx_frac=logo_cx,
        )))
    frames.extend(_hold(frames[-1], hold))
    logo_hold = frames[-1]

    # --- Crossfade into D4 ---
    d4_bare = plot_d4(show_threshold=False)
    d4_bare_e = emit(d4_bare)
    for i in range(max(1, n_fade)):
        u = smooth(float(i + 1) / max(n_fade, 1))
        frames.append(ch5_crossfade_images(logo_hold, d4_bare_e, u))
    frames.extend(_hold(d4_bare_e, hold))

    # Best threshold (thick dashed), no equation legend
    d4_map = plot_d4(show_threshold=True, thr_lw=lw_thick)
    add(d4_map, n=hold)

    # Rotating cuts one-at-a-time; MAP stays thick dashed; cuts thin dashed
    for cws, cwe, cbb in rotating:
        extra = [_ch5_dashed_threshold_extra(cws, cwe, cbb, linewidth=lw_thin)]
        add(plot_d4(show_threshold=True, extra=extra, thr_lw=lw_thick), n=n_cut_hold)

    # --- Fade out ---
    last_d4 = frames[-1]
    white = Image.new("RGB", last_d4.size, (255, 255, 255))
    for i in range(max(1, n_fade)):
        u = smooth(float(i + 1) / max(n_fade, 1))
        frames.append(ch5_crossfade_images(last_d4, white, u))
    frames.extend(_hold(white, max(1, hold // 2)))

    # --- Fade in D1 + MAP (normal thickness) ---
    d1_map2 = plot_d1(show_threshold=True, legend=False, thr_lw=lw_norm)
    d1_map2_e = emit(d1_map2)
    for i in range(max(1, n_fade)):
        u = smooth(float(i + 1) / max(n_fade, 1))
        frames.append(ch5_crossfade_images(white, d1_map2_e, u))
    frames.extend(_hold(d1_map2_e, hold))

    # Parallel dashed family one-at-a-time (like D4 cuts); MAP stays thick
    for extra in par_extras:
        add(
            plot_d1(show_threshold=True, extra=[extra], legend=False, thr_lw=lw_thick),
            n=n_cut_hold,
        )
    frames.extend(_hold(frames[-1], n_par))

    # Drop parallels; MAP back to normal thickness
    d1_map_norm = plot_d1(show_threshold=True, legend=False, thr_lw=lw_norm)
    add(d1_map_norm, n=hold)

    # Wide 2D → duo resize first (no 3D), then empty Belief axes appear
    for i in range(max(1, n_layout)):
        u = smooth(float(i) / max(n_layout - 1, 1))
        frames.append(emit(_frame_d1_wide_to_duo_empty3d(
            u, ws=map1_ws, we=map1_we, bb=map1_bb,
            threshold_linewidth=lw_norm, threshold_legend=False,
            show_3d_u=0.0,
        )))
    frames.extend(_hold(frames[-1], hold))
    duo_no3d = frames[-1]
    duo_with3d = emit(_frame_d1_wide_to_duo_empty3d(
        1.0, ws=map1_ws, we=map1_we, bb=map1_bb,
        threshold_linewidth=lw_norm, threshold_legend=False,
        show_3d_u=1.0,
    ))
    for i in range(max(1, n_fade)):
        u = smooth(float(i + 1) / max(n_fade, 1))
        frames.append(ch5_crossfade_images(duo_no3d, duo_with3d, u))
    frames.extend(_hold(frames[-1], hold))
    morph_end = frames[-1]

    # Landscape reveal like ch5_55 (diagonal wipe on D1 posterior belief)
    cfg = ch5_prior_landscape.ch5_grid_landscape_config()
    fk = dict(CH5_HQ_LAND_FRAME_KW)
    datasets = _ch5_landscape_grid_datasets()
    final = ch5_prior_landscape._ch5_landscape_grid_final_pack(datasets, config=cfg)
    per_key = final["per_key"]
    pk = final["prior_kind"]
    z_color_lim = final["z_color_lim"]
    extra = per_key["D1"]
    mesh = extra["mesh_pack"]
    z_lim = extra["z_lim"]
    el_land = ch5_prior_landscape._ch5_hq_land_elev()
    az_land = float(_g("CH3_LIK_W12_CT_AZIM"))
    wz, ez, bz = CH5_KNOB_ZERO
    reveal_origin = str(CH5_LL_OVERLAY_REVEAL_ORIGIN)

    def _land_frame(lrev: float, *, marker: bool = False, surface: bool = True):
        return ch5_prior_landscape._ch5_frame_lik_w12_belief(
            study1, exam1, y1,
            float(wz), float(ez), float(bz),
            mesh_pack=mesh,
            z_lim=z_lim,
            prior_kind=pk,
            z_color_lim=z_color_lim,
            elev=el_land,
            azim=az_land,
            landscape_reveal=float(lrev),
            landscape_reveal_origin=reveal_origin,
            show_curves=False,
            marker=bool(marker),
            marker_ws=float(extra["marker_ws"]),
            marker_we=float(extra["marker_we"]),
            marker_z=float(extra["marker_z"]),
            here_annotation=bool(marker),
            here_label="most plausible line",
            here_text_color="white",
            show_threshold=True,
            threshold_ws=float(map1_ws),
            threshold_we=float(map1_we),
            threshold_bb=float(map1_bb),
            threshold_label="",
            threshold_legend_dark=False,
            z_lik_ref=float(extra["z_ref"]),
            show_surface=bool(surface) and float(lrev) > 1e-6,
            surface_grid=bool(surface) and float(lrev) > 1e-6,
            belief_surface_alpha=float(CH5_BELIEF_SURFACE_ALPHA),
            frame_kwargs=fk,
            **dict(CH5_KNOBS_UNSET_FRAME_KW),
        )

    with ch5_prior_landscape._landscape_render_context(cfg.dpi):
        empty_duo = emit(_land_frame(0.0, marker=False, surface=False))
        for i in range(max(1, n_fade)):
            u = smooth(float(i + 1) / max(n_fade, 1))
            frames.append(ch5_crossfade_images(morph_end, empty_duo, u))
        frames.extend(_hold(empty_duo, max(1, hold // 2)))

        for i in range(max(1, n_land)):
            u = smooth(float(i) / max(n_land - 1, 1))
            frames.append(emit(_land_frame(u, marker=u > 0.92, surface=True)))
        frames.extend(_hold(frames[-1], hold))

    return frames


def build_ch5_d1_best_line_howithink_d4_cuts(clip_id):
    return _build_d1_best_line_howithink_d4_cuts_story(clip_id)


def _build_credible_voxel_story(key, clip_id, *, prior_kind="gaussian"):
    """Belief cloud → 90° camera spin → shadow/collapse per axis → 360° orbit."""
    frames = []
    n_fill = int(CH5_VOXEL_N_FILL)
    n_rot = int(CH5_VOXEL_N_ROT)
    n_shadow = int(CH5_VOXEL_N_PROJ_SHADOW)
    n_collapse = int(CH5_VOXEL_N_PROJ_COLLAPSE)
    hold = _draft_range(6, 3)
    proj_hold = _draft_range(4, 2)

    for _ in range(hold):
        plot = _frame_credible_voxel(
            key, prior_kind=prior_kind, voxel_fill_u=0.0,
        )
        frames.append(_finish(plot, clip_id, show_legend=True))

    for tv in np.linspace(0.0, 1.0, n_fill, endpoint=True):
        u = _g("ch3_knob_smoothstep")(float(tv))
        plot = _frame_credible_voxel(
            key, prior_kind=prior_kind, voxel_fill_u=float(u),
        )
        frames.append(_finish(plot, clip_id, show_legend=True))

    for _ in range(hold):
        plot = _frame_credible_voxel(
            key, prior_kind=prior_kind, voxel_fill_u=1.0,
        )
        frames.append(_finish(plot, clip_id, show_legend=True))

    for tv in np.linspace(0.0, 1.0, n_rot, endpoint=True):
        u = _g("ch3_knob_smoothstep")(float(tv))
        plot = _frame_credible_voxel(
            key, prior_kind=prior_kind, voxel_fill_u=1.0, cam_azim_u=float(u),
        )
        frames.append(_finish(plot, clip_id, show_legend=True))

    for _ in range(hold):
        plot = _frame_credible_voxel(
            key, prior_kind=prior_kind, voxel_fill_u=1.0, cam_azim_u=1.0,
        )
        frames.append(_finish(plot, clip_id, show_legend=True))

    done = ()
    for axis in ("el", "st", "b"):
        for tv in np.linspace(0.0, 1.0, n_shadow, endpoint=True):
            u = _g("ch3_knob_smoothstep")(float(tv))
            plot = _frame_credible_voxel(
                key, prior_kind=prior_kind, voxel_fill_u=1.0, cam_azim_u=1.0,
                proj_states=_ch5_proj_states(
                    axis, shadow_u=u, collapse_u=0.0, done=done,
                ),
            )
            frames.append(_finish(plot, clip_id, show_legend=True))

        for _ in range(proj_hold):
            plot = _frame_credible_voxel(
                key, prior_kind=prior_kind, voxel_fill_u=1.0, cam_azim_u=1.0,
                proj_states=_ch5_proj_states(
                    axis, shadow_u=1.0, collapse_u=0.0, done=done,
                ),
            )
            frames.append(_finish(plot, clip_id, show_legend=True))

        for tv in np.linspace(0.0, 1.0, n_collapse, endpoint=True):
            u = _g("ch3_knob_smoothstep")(float(tv))
            plot = _frame_credible_voxel(
                key, prior_kind=prior_kind, voxel_fill_u=1.0, cam_azim_u=1.0,
                proj_states=_ch5_proj_states(
                    axis, shadow_u=1.0, collapse_u=u, done=done,
                ),
            )
            frames.append(_finish(plot, clip_id, show_legend=True))

        done = done + (axis,)
        for _ in range(hold):
            plot = _frame_credible_voxel(
                key, prior_kind=prior_kind, voxel_fill_u=1.0, cam_azim_u=1.0,
                proj_states=_ch5_proj_states(done=done),
            )
            frames.append(_finish(plot, clip_id, show_legend=True))

    n_orbit = int(CH5_VOXEL_N_ORBIT)
    orbit_spin = float(_g("CH3_LIK_3D_CAM_PATH_ROT")) + float(CH5_VOXEL_ORBIT_DEG)
    orbit_u0 = float(_g("CH3_LIK_3D_CAM_PATH_ROT")) / orbit_spin
    finale_proj = _ch5_proj_states(done=done)
    for tv in np.linspace(0.0, 1.0, n_orbit, endpoint=True):
        u = orbit_u0 + (1.0 - orbit_u0) * _g("ch3_knob_smoothstep")(float(tv))
        plot = _frame_credible_voxel(
            key, prior_kind=prior_kind, voxel_fill_u=1.0,
            cam_azim_u=float(u), cam_spin_deg=orbit_spin,
            proj_states=finale_proj,
        )
        frames.append(_finish(plot, clip_id, show_legend=True))

    for _ in range(hold):
        plot = _frame_credible_voxel(
            key, prior_kind=prior_kind, voxel_fill_u=1.0,
            cam_azim_u=1.0, cam_spin_deg=orbit_spin,
            proj_states=finale_proj,
        )
        frames.append(_finish(plot, clip_id, show_legend=True))
    return frames


def _frame_decomposition(key="D1", *, phase_u=0.0):
    pack = _PACKS[key]
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    u = float(np.clip(phase_u, 0.0, 1.0))
    b_fixed = float(CH5_W12_B_FIXED)
    W1, W2 = ch5_w12_mesh()
    B = np.full_like(W1, b_fixed)
    dlo1, dhi1, dlo2, dhi2, _, _ = CH5_VIEW_BOUNDS
    lp = ch5_prior_w12_log_flat(W1, W2, B, kind="gaussian")
    log_lik = ch5_log_likelihood_grid(study, exam, y, W1, W2, B, nll_fn=_g("_ch3_nll_sum_on_flat_grid"))
    log_post = lp + log_lik
    prior_pdf = ch5_belief_w12_pdf(
        ch5_prior_w12_log_flat(W1, W2, B, kind="gaussian"),
        w1_lo=dlo1, w1_hi=dhi1, w2_lo=dlo2, w2_hi=dhi2,
    )
    post_pdf = ch5_belief_w12_pdf(
        log_post.reshape(W1.shape), w1_lo=dlo1, w1_hi=dhi1, w2_lo=dlo2, w2_hi=dhi2,
    )
    ll_fin = log_lik[np.isfinite(log_lik)]
    ll_max = float(np.max(ll_fin)) if ll_fin.size else 0.0
    lik_peak = float(ch5_prior_w12_belief_peak(kind="gaussian"))
    lik_display = np.exp(log_lik - ll_max) * lik_peak
    if u < 1.0 / 3.0:
        density = prior_pdf
    elif u < 2.0 / 3.0:
        t = (u - 1.0 / 3.0) * 3.0
        density = (1.0 - t) * prior_pdf + t * lik_display
    else:
        t = (u - 2.0 / 3.0) * 3.0
        density = (1.0 - t) * lik_display + t * post_pdf
    density = ch5_clip_belief_height(density, prior_kind="gaussian")
    ws, we, bb = pack["ws"], pack["we"], pack["bb"]
    fig, ax_data, ax3d, axes_k = _g("ch4_figure_duo_weight3d")()
    _draw_roster_ax(ax_data, pack, show_line=True)
    knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
    _g("ch3_draw_knob_row")(
        fig, axes_k, ws, we, bb, "st", knob_rgbs, canvas_sides,
        rot_strip_deg=0.0, strip_scale=1.0,
        knob_rots=_g("ch3_k1_knob_rots_at")(ws, we, bb), knob_scales=[1.0, 1.0, 1.0], ax_data=ax_data,
    )
    _style_ax3d_mini(ax3d, prior_kind="gaussian")
    _draw_density_surface(ax3d, W1, W2, density, colored=True, prior_kind="gaussian")
    return _fig_to_plot(fig)


def _build_ct_story(key, clip_id, *, n_show=None, prior_kind="gaussian"):
    """Ch4-style CT scan: b hold → st → el → b sweeps with pivots."""
    frames = []
    bounds = CH5_VIEW_BOUNDS
    ct_axes = _g("CH3_LIK_CT_AXES")
    pivot_map = dict(_g("CH3_LIK_CT_PIVOTS"))
    n_hold = int(_g("CH3_LIK_CT_N_HOLD"))
    kw: dict[str, Any] = {"prior_kind": str(prior_kind)}
    if n_show is not None:
        kw["n_show"] = n_show

    for _ in range(n_hold):
        plot = _frame_ct_single(key, sweep_axis="b", plane_val=float(bounds[4]), **kw)
        frames.append(_finish(plot, clip_id, show_legend=True))

    for i, axis in enumerate(ct_axes):
        if i > 0:
            prev = ct_axes[i - 1]
            nxt = pivot_map.get(prev)
            if nxt == axis:
                for tv in np.linspace(0.0, 1.0, CH5_CT_N_PIVOT, endpoint=True):
                    plot = _frame_ct_single(
                        key, pivot_from=prev, pivot_to=axis, pivot_u=float(tv), **kw,
                    )
                    frames.append(_finish(plot, clip_id, show_legend=True))
        lo, hi = _g("_ch4_ct_axis_limits")(axis, bounds)
        for tv in np.linspace(0.0, 1.0, CH5_CT_N_SWEEP, endpoint=True):
            u = _g("ch3_knob_smoothstep")(float(tv))
            val = lo + u * (hi - lo)
            plot = _frame_ct_single(key, sweep_axis=axis, plane_val=val, **kw)
            frames.append(_finish(plot, clip_id, show_legend=True))
    frames.extend(_hold(frames[-1]))
    return frames


def _build_ct_grid_story(clip_id, active_keys=None, *, prior_kind="gaussian"):
    """2×2 CT grid through the same Ch4-style sweep choreography."""
    keys = list(CH5_DATASET_KEYS if active_keys is None else active_keys)
    pk = str(prior_kind)
    frames = []
    bounds = CH5_VIEW_BOUNDS
    ct_axes = _g("CH3_LIK_CT_AXES")
    pivot_map = dict(_g("CH3_LIK_CT_PIVOTS"))
    n_hold = int(_g("CH3_LIK_CT_N_HOLD"))

    for _ in range(n_hold):
        img = _frame_ct_grid_2x2(keys, sweep_axis="b", plane_val=float(bounds[4]), prior_kind=pk)
        frames.append(_finish(img, clip_id, show_legend=True))

    for i, axis in enumerate(ct_axes):
        if i > 0:
            prev = ct_axes[i - 1]
            nxt = pivot_map.get(prev)
            if nxt == axis:
                for tv in np.linspace(0.0, 1.0, CH5_CT_N_PIVOT, endpoint=True):
                    img = _frame_ct_grid_2x2(
                        keys, pivot_from=prev, pivot_to=axis, pivot_u=float(tv), prior_kind=pk,
                    )
                    frames.append(_finish(img, clip_id, show_legend=True))
        lo, hi = _g("_ch4_ct_axis_limits")(axis, bounds)
        for tv in np.linspace(0.0, 1.0, CH5_CT_N_SWEEP, endpoint=True):
            u = _g("ch3_knob_smoothstep")(float(tv))
            val = lo + u * (hi - lo)
            img = _frame_ct_grid_2x2(keys, sweep_axis=axis, plane_val=val, prior_kind=pk)
            frames.append(_finish(img, clip_id, show_legend=True))
    frames.extend(_hold(frames[-1]))
    return frames


def _build_sequential_w12(clip_id, *, key="D1", prior_kind="gaussian"):
    pack = _PACKS[key]
    order = pack["order"]
    frames = []
    for n in range(0, len(order) + 1):
        plot = _frame_duo_posterior(
            key, n_show=n, prior_kind=prior_kind, mono=True, colored=False,
        )
        right = [{"label": r"$n$", "text": rf"$n={n}$", "bold_lhs": True, "role": "ndata"}]
        frames.append(_finish(plot, clip_id, right_blocks=right))
        frames.extend(_hold(frames[-1], CH5_N_SEQ_HOLD))
    frames.extend(_hold(frames[-1]))
    return frames


def _build_sequential_ct(clip_id, *, key="D1", prior_kind="gaussian"):
    """Sequential CT: b-sweep while points arrive; full Ch4 story once complete."""
    pack = _PACKS[key]
    order = pack["order"]
    bounds = CH5_VIEW_BOUNDS
    _, _, _, _, dlob, dhib = bounds
    n_sweep = _draft_range(10, 4)
    frames = []
    for n in range(0, len(order) + 1):
        if n == len(order):
            frames.extend(_build_ct_story(key, clip_id, n_show=n, prior_kind=prior_kind))
        else:
            for tv in np.linspace(0.0, 1.0, n_sweep, endpoint=True):
                val = float(dlob + float(tv) * (dhib - dlob))
                plot = _frame_ct_single(
                    key, sweep_axis="b", plane_val=val, n_show=n, prior_kind=prior_kind,
                )
                frames.append(_finish(plot, clip_id, show_legend=True))
            frames.extend(_hold(frames[-1], max(1, CH5_N_SEQ_HOLD // 2)))
    return frames


def _build_sequential_d1(clip_id):
    return _build_sequential_w12(clip_id, key="D1", prior_kind="gaussian")


def _build_ct_d1(clip_id):
    return _build_ct_story("D1", clip_id)


# --- Export builders (≥1 s at full quality; sequential ch5_01 … ch5_41) ---

def build_ch5_02_sequential_bars(clip_id):
    return _build_sequential_bars_story(clip_id)


def build_ch5_03e(clip_id):
    return _build_sequential_w12(clip_id, key="D1", prior_kind="gaussian")


def build_ch5_03f(clip_id):
    return _build_sequential_w12(clip_id, key="D1", prior_kind="uniform")


def build_ch5_03g(clip_id):
    return _build_sequential_ct(clip_id, key="D1", prior_kind="gaussian")


def _build_prior_landscape(clip_id, *, prior_kind: str):
    raw = ch5_prior_landscape.ch5_build_prior_w12_landscape_frames(prior_kind)
    return [_finish(fr, clip_id) for fr in raw]


def build_ch5_04b(clip_id):
    return _build_prior_landscape(clip_id, prior_kind="gaussian")


def build_ch5_04c(clip_id):
    return _build_prior_landscape(clip_id, prior_kind="uniform")


def build_ch5_04d(clip_id):
    return _build_sequential_d1(clip_id)


def build_ch5_04e(clip_id):
    frames = []
    for tv in np.linspace(0.0, 1.0, CH5_N_DECOMP, endpoint=True):
        u = _g("ch3_knob_smoothstep")(float(tv))
        plot = _frame_decomposition("D1", phase_u=u)
        frames.append(_finish(plot, clip_id, show_legend=True))
    return frames + _hold(frames[-1])


def _build_per_dataset_pack(key, clip_id, part, *, prior_kind="gaussian"):
    if part == "squish":
        frames = []
        for u in np.linspace(0, 1, _draft_range(10, 4)):
            plot = _frame_duo_posterior(key, squish_u=float(u), prior_kind=prior_kind)
            frames.append(_finish(plot, clip_id, show_legend=True))
        return frames + _hold(frames[-1])
    if part == "ct":
        return _build_ct_story(key, clip_id, prior_kind=prior_kind)
    raise ValueError(part)


def _make_dataset_landscape_builder(key: str, part: str, *, prior_kind: str):
    def _builder(clip_id: str):
        return _build_per_dataset_pack(key, clip_id, part, prior_kind=prior_kind)
    return _builder


def build_ch5_06_montage(clip_id):
    frames = []
    for keys in (["D1"], ["D1", "D2"], ["D1", "D2", "D3"], list(CH5_DATASET_KEYS)):
        img = _frame_duo_grid_2x2(keys)
        frames.append(_finish(img, clip_id, show_legend=True))
        frames.extend(_hold(frames[-1], 1 if _CH3_DRAFT else 3))
    return frames


def build_ch5_07_duo_grid(clip_id):
    frames = []
    for keys in (["D1"], ["D1", "D2"], ["D1", "D2", "D3"], CH5_DATASET_KEYS):
        img = _frame_grid_duo(keys)
        frames.append(_finish(img, clip_id, show_legend=True))
        frames.extend(_hold(frames[-1]))
    return frames


def build_ch5_07_duo_grid_uniform(clip_id):
    frames = []
    for keys in (["D1"], ["D1", "D2"], ["D1", "D2", "D3"], CH5_DATASET_KEYS):
        img = _frame_grid_duo(keys, prior_kind="uniform")
        frames.append(_finish(img, clip_id, show_legend=True))
        frames.extend(_hold(frames[-1]))
    return frames


def build_ch5_07b_ct_grid(clip_id):
    return _build_ct_grid_story(clip_id, CH5_DATASET_KEYS)


def build_ch5_07b_ct_grid_uniform(clip_id):
    """ch5_29: 47+ surfaces → squish @ b=0 → continuous HQ CT scan."""
    return _build_uniform_landscape_squish_ct_story(clip_id)


def build_ch5_08a_credible_fill(clip_id):
    frames = []
    pack = _PACKS["D1"]
    masses = np.linspace(0.5, pack["meta"]["credible_target"], _draft_range(12, 5))
    for m in masses:
        plot = _frame_duo_posterior("D1", credible_mass=float(m))
        frames.append(_finish(plot, clip_id, show_legend=True))
    return frames + _hold(frames[-1])


def build_ch5_08c_axis_D1(clip_id):
    return _build_credible_voxel_story("D1", clip_id, prior_kind="gaussian")


def build_ch5_10_credible_4x4(clip_id):
    frames = []
    for rows in (["D1"], ["D1", "D2"], ["D1", "D2", "D3"], CH5_DATASET_KEYS):
        img = _frame_credible_4x4(active_rows=rows)
        frames.append(_finish(img, clip_id, show_legend=True))
        frames.extend(_hold(frames[-1], 1 if _CH3_DRAFT else 4))
    return frames


def build_ch5_12b_ct_D2(clip_id):
    return _build_per_dataset_pack("D2", clip_id, "ct")


def build_ch5_dataset_2d_map(clip_id):
    return _build_dataset_2d_map_story(clip_id)


def _build_credibility_prior_intro_story(clip_id, *, compose_id: str = "ch5_01"):
    """
    Pre-landscape credibility / prior beat (full-bleed 2D only).

    Likelihood thickness race → equal-likelihood fan → D4 tease → empty axes →
    domain veto (absurd ST/EL) → unequal prior thickness → data refine →
    morph into ch5_01 opening (same handoff as intro_to_01).
    """
    frames: list = []
    pack = _PACKS["D1"]
    study = np.asarray(pack["study"], dtype=np.float64)
    exam = np.asarray(pack["exam"], dtype=np.float64)
    y = np.asarray(pack["y"], dtype=np.int64)
    point_order = _ch5_d1_sequential_point_order()
    n_pts = len(point_order)
    full_mask = np.ones(len(y), dtype=bool)
    empty_mask = np.zeros(len(y), dtype=bool)
    xl0, yl0 = ch5_plot_limits("D1")
    # Absurd domain for the 1M-hours / long-exam beat.
    xl_abs = (0.0, 1.0e6)
    yl_abs = (0.0, 24.0)
    smooth = _g("ch3_knob_smoothstep")

    hold = _draft_range(8, 2)
    hold_short = _draft_range(5, 2)
    n_fade = _draft_range(12, 4)
    n_fan = _draft_range(7, 4)
    n_fan_hold = _draft_range(6, 2)
    n_zoom = _draft_range(18, 5)
    n_d4_cuts = _draft_range(6, 3)
    n_d4_hold = _draft_range(5, 2)
    n_count = _draft_range(4, 2)
    n_duo = _draft_range(CH5_INTRO_N_DUO_MORPH, 6)
    n_layout = _draft_range(CH5_INTRO_N_LAYOUT_MORPH, 6)
    n_panel = _draft_range(CH5_INTRO_N_PANEL_MORPH, 5)
    plot_rect = _g("CH4_LIK_PLOT_START_RECT")

    lines5 = list(CH5_FIVE_LINES)
    colors5 = list(CH5_FIVE_LINE_COLORS)
    # Domain-absurd hypotheses (only "sensible" when axes are huge).
    absurd_lines = (
        (1.0, 0.0, -5.0e5),   # vertical-ish cut near ST ≈ 5e5
        (0.0, 1.0, -18.0),    # horizontal cut near EL ≈ 18h
        (2.0, -0.05, -1.0e6), # near-vertical extreme slope
    )
    absurd_colors = ("#888888", "#888888", "#888888")

    # Prior preference among five lines before any data (favor moderate middle line).
    prior_w = np.array([0.12, 0.40, 0.12, 0.18, 0.18], dtype=np.float64)
    prior_w = prior_w / float(prior_w.sum())

    def emit(img):
        return _ch5_finish_duo_export(img, clip_id)

    def add_plot(img, *, n=None):
        out = emit(img)
        frames.append(out)
        if n is not None and int(n) > 1:
            frames.extend(_hold(out, int(n) - 1))

    def panel(*, mask, lines, colors, weights=None, xl=None, yl=None, strike=False):
        xl_use = xl0 if xl is None else xl
        yl_use = yl0 if yl is None else yl
        lines = list(lines)
        colors = list(colors)
        wts = None if weights is None else np.asarray(weights, dtype=np.float64)

        def _draw(ax, _xl, _yl):
            _draw_hypothesis_lines_on_ax(
                ax, _xl, _yl, lines, colors,
                n_lines=len(lines), line_weights=wts,
            )
            if strike:
                # Red veto in axes fraction (scale-independent).
                ax.plot(
                    [0.42, 0.58], [0.42, 0.58],
                    transform=ax.transAxes, c="#c62828", lw=3.4, alpha=0.92, zorder=6,
                    clip_on=False,
                )
                ax.plot(
                    [0.42, 0.58], [0.58, 0.42],
                    transform=ax.transAxes, c="#c62828", lw=3.4, alpha=0.92, zorder=6,
                    clip_on=False,
                )

        return _frame_plot_2d_points(
            study, exam, y, mask=mask, after_draw=_draw, xl=xl_use, yl=yl_use,
        )

    # --- 1. Likelihood comparison on D1 ---
    lik0 = _ch5_line_likelihood_weights(study, exam, y, lines5)
    winner = int(np.argmax(lik0))
    add_plot(panel(mask=full_mask, lines=lines5, colors=colors5, weights=lik0), n=hold)

    # --- 2. Low-lik losers fade ---
    fade_w = lik0.copy()
    fade_w = fade_w / max(float(fade_w.max()), 1e-12)
    fade_w = np.where(np.arange(len(fade_w)) == winner, 1.0, fade_w * 0.15)
    add_plot(panel(mask=full_mask, lines=lines5, colors=colors5, weights=fade_w), n=hold_short)
    # Briefly flash a bad line, then return to faded losers.
    bad = int(np.argmin(lik0))
    flash = np.full(len(lines5), 0.08, dtype=np.float64)
    flash[winner] = 1.0
    flash[bad] = 0.85
    add_plot(panel(mask=full_mask, lines=lines5, colors=colors5, weights=flash), n=hold_short)
    add_plot(panel(mask=full_mask, lines=lines5, colors=colors5, weights=fade_w), n=hold)

    # --- 3–4. Equal-likelihood parallel fan (undercut uniqueness) ---
    w_ws, w_we, w_bb = lines5[winner]
    fan = _ch5_parallel_line_family(w_ws, w_we, w_bb, n=n_fan, bias_span=1.55)
    fan_colors = ["#6a6a6a"] * len(fan)
    # Spawn one-at-a-time then equalize.
    for k in range(1, len(fan) + 1):
        wts = np.ones(k, dtype=np.float64)
        add_plot(
            panel(mask=full_mask, lines=fan[:k], colors=fan_colors[:k], weights=wts),
            n=n_fan_hold,
        )
    # --- 5. How many? count-in pulse then hold ---
    equal = np.ones(len(fan), dtype=np.float64)
    for i in range(len(fan)):
        pulse = equal.copy()
        pulse[i] = 2.2
        add_plot(
            panel(mask=full_mask, lines=fan, colors=fan_colors, weights=pulse),
            n=n_count,
        )
    add_plot(
        panel(mask=full_mask, lines=fan, colors=fan_colors, weights=equal),
        n=hold,
    )

    # --- short D4 tease: many cuts look equally fine ---
    pack4 = _PACKS["D4"]
    study4 = np.asarray(pack4["study"], dtype=np.float64)
    exam4 = np.asarray(pack4["exam"], dtype=np.float64)
    y4 = np.asarray(pack4["y"], dtype=np.int64)
    mask4 = np.ones(len(y4), dtype=bool)
    xl4, yl4 = ch5_plot_limits("D4")
    cx4, cy4 = CH5_D4_CENTER
    cut_scale = max(float(np.hypot(w_ws, w_we)), 0.85)
    rotating = _ch5_d4_rotating_cuts(n_d4_cuts, scale=cut_scale)

    def panel_d4(lines, weights=None):
        cols = ["#6a6a6a"] * len(lines)
        wts = None if weights is None else np.asarray(weights, dtype=np.float64)

        def _draw(ax, _xl, _yl):
            _draw_hypothesis_lines_on_ax(
                ax, _xl, _yl, lines, cols, n_lines=len(lines), line_weights=wts,
            )

        return _frame_plot_2d_points(
            study4, exam4, y4, mask=mask4, after_draw=_draw, xl=xl4, yl=yl4,
        )

    d1_hold = frames[-1]
    d4_first = emit(panel_d4([rotating[0]], weights=np.ones(1)))
    for i in range(max(1, n_fade)):
        u = smooth(float(i + 1) / max(n_fade, 1))
        frames.append(ch5_crossfade_images(d1_hold, d4_first, u))
    for cut in rotating:
        add_plot(panel_d4([cut], weights=np.ones(1)), n=n_d4_hold)
    # Show several at once equally.
    add_plot(panel_d4(rotating, weights=np.ones(len(rotating))), n=hold)

    # Crossfade back to empty D1 axes with candidate lines.
    candidates = list(CH5_THREE_LINES) + list(CH5_INTRO_EXTRA_LINES[:2])
    cand_colors = list(CH5_THREE_LINE_COLORS) + list(CH5_FIVE_LINE_COLORS[3:5])
    equal_cand = np.ones(len(candidates), dtype=np.float64)
    empty_equal = emit(panel(
        mask=empty_mask, lines=candidates, colors=cand_colors, weights=equal_cand,
    ))
    last_d4 = frames[-1]
    for i in range(max(1, n_fade)):
        u = smooth(float(i + 1) / max(n_fade, 1))
        frames.append(ch5_crossfade_images(last_d4, empty_equal, u))
    frames.extend(_hold(empty_equal, hold))

    # --- 6–7. Domain prior: zoom absurd → veto → snap back → unequal prior ---
    zoomed_lines = list(candidates) + list(absurd_lines)
    zoomed_colors = list(cand_colors) + list(absurd_colors)
    zoomed_w = np.concatenate([
        np.ones(len(candidates), dtype=np.float64) * 0.55,
        np.ones(len(absurd_lines), dtype=np.float64),
    ])
    for i in range(max(1, n_zoom)):
        u = float(i) / max(n_zoom - 1, 1)
        xl, yl = _ch5_lerp_xy_limits(xl0, yl0, xl_abs, yl_abs, u)
        add_plot(panel(
            mask=empty_mask, lines=zoomed_lines, colors=zoomed_colors,
            weights=zoomed_w, xl=xl, yl=yl,
        ))
    # Strike / fade absurd lines at full zoom.
    add_plot(panel(
        mask=empty_mask, lines=zoomed_lines, colors=zoomed_colors,
        weights=zoomed_w, xl=xl_abs, yl=yl_abs, strike=True,
    ), n=hold_short)
    veto_w = zoomed_w.copy()
    veto_w[len(candidates):] = 0.05
    add_plot(panel(
        mask=empty_mask, lines=zoomed_lines, colors=zoomed_colors,
        weights=veto_w, xl=xl_abs, yl=yl_abs, strike=True,
    ), n=hold_short)
    # Snap back; absurd gone; prior thicknesses unequal.
    for i in range(max(1, n_zoom)):
        u = float(i) / max(n_zoom - 1, 1)
        xl, yl = _ch5_lerp_xy_limits(xl_abs, yl_abs, xl0, yl0, u)
        # Drop absurd lines as we return.
        show_abs = u < 0.55
        if show_abs:
            lw = list(candidates) + list(absurd_lines)
            lc = list(cand_colors) + list(absurd_colors)
            ww = np.concatenate([prior_w[: len(candidates)], np.full(len(absurd_lines), 0.04)])
        else:
            lw, lc, ww = candidates, cand_colors, prior_w[: len(candidates)]
        add_plot(panel(
            mask=empty_mask, lines=lw, colors=lc, weights=ww, xl=xl, yl=yl,
        ))
    add_plot(panel(
        mask=empty_mask, lines=candidates, colors=cand_colors, weights=prior_w[: len(candidates)],
    ), n=hold)

    # --- 8. Data refines prior (likelihood × prior thicknesses) ---
    for n in range(1, n_pts + 1):
        mask = np.zeros(len(y), dtype=bool)
        for j in point_order[:n]:
            mask[int(j)] = True
        sn, en, yn = study[mask], exam[mask], y[mask]
        lik = _ch5_line_likelihood_weights(sn, en, yn, candidates)
        # Bayes-lite: prior × likelihood, renormalized (still discrete lines only).
        post = prior_w[: len(candidates)] * lik
        post = post / max(float(post.sum()), 1e-12)
        add_plot(
            panel(mask=mask, lines=candidates, colors=cand_colors, weights=post),
            n=1 if _CH3_DRAFT else 2,
        )
    final_w = prior_w[: len(candidates)] * _ch5_line_likelihood_weights(
        study, exam, y, candidates,
    )
    final_w = final_w / max(float(final_w.sum()), 1e-12)
    winner_f = int(np.argmax(final_w))
    # Survivor emphasis.
    surv = np.full(len(candidates), 0.08, dtype=np.float64)
    surv[winner_f] = 1.0
    add_plot(panel(mask=full_mask, lines=candidates, colors=cand_colors, weights=surv), n=hold)

    # --- Handoff: wipe → duo morph → ch5_01 opening (intro_to_01 beats 7–11) ---
    add_plot(panel(mask=empty_mask, lines=[], colors=[], weights=None), n=hold_short)

    global _CH5_INTRO_DUO_RECTS
    _CH5_INTRO_DUO_RECTS = None
    for tv in np.linspace(0.0, 1.0, n_duo, endpoint=True):
        u = float(smooth(float(tv)))
        frames.append(emit(_frame_ch5_intro_duo_plot(
            u, u, u, mask=empty_mask, n_lines=0, n_bars=0,
        )))
    frames.extend(_hold(frames[-1], hold_short))
    plot0 = frames[-1]

    def add_comp(img, *, layout_u=1.0, panel_u=1.0, title_u=1.0, write_u=1.0, plot_start=None):
        empty = np.array([], dtype=np.float64)
        empty_y = np.array([], dtype=np.int64)
        right_blocks = _ch5_exact_values_blocks(empty, empty, empty_y)
        right_prog = _ch5_seq_right_progress(n_lines=0, n_prior_vals=0)
        kw = dict(
            layout_u=layout_u,
            panel_u=panel_u,
            title_write_progress=title_u,
            write_progress=write_u,
        )
        if plot_start is not None:
            kw["plot_start_rect"] = plot_start
        frames.append(_ch5_finish_bars(
            img, compose_id,
            right_blocks=right_blocks,
            progress_override={"right": right_prog},
            **kw,
        ))

    # plot0 is already export-composed; need raw plot for layout morph.
    # Rebuild a raw empty duo-end frame for the tutorial shell morph.
    raw0 = _frame_ch5_intro_duo_plot(1.0, 1.0, 1.0, mask=empty_mask, n_lines=0, n_bars=0)
    for tv in np.linspace(0.0, 1.0, n_layout, endpoint=True):
        u = float(smooth(float(tv)))
        add_comp(raw0, layout_u=u, panel_u=0.0, title_u=0.0, write_u=0.0, plot_start=plot_rect)
    for tv in np.linspace(0.0, 1.0, n_panel, endpoint=True):
        u = float(smooth(float(tv)))
        add_comp(raw0, layout_u=1.0, panel_u=u, title_u=u, write_u=1.0, plot_start=plot_rect)

    opening = _ch5_01_opening_frame(compose_id)
    frames.extend(_hold(opening, hold))
    frames.extend(_hold(opening))
    return frames


def build_ch5_credibility_prior_intro(clip_id):
    return _build_credibility_prior_intro_story(clip_id, compose_id="ch5_01")


def build_ch5_intro_to_01(clip_id):
    return _build_intro_to_01_story(clip_id, compose_id="ch5_01")


def build_ch5_tutorial_to_prior(clip_id):
    return _build_tutorial_to_prior_landscape_story(clip_id, compose_id="ch5_01")


def build_ch5_tutorial_to_prior_stem_view(clip_id):
    """ch5_44 with ch5_58 fixed end-camera (no W1/W2 view dance)."""
    return _build_tutorial_to_prior_landscape_story(
        clip_id, compose_id="ch5_01", fixed_end_camera=True,
    )


def build_ch5_sequential_posterior_landscape(clip_id):
    return _build_sequential_posterior_landscape_story(clip_id)


def build_ch5_uniform_prior_to_sequential(clip_id):
    return _build_uniform_prior_to_sequential_story(clip_id)


def build_ch5_uniform_landscape_grid(clip_id):
    return _build_uniform_landscape_grid_story(clip_id)


def build_ch5_uniform_landscape_grid_zoom(clip_id):
    return _build_uniform_landscape_grid_zoom_story(clip_id)


def build_ch5_uniform_landscape_grid_focus(clip_id):
    return _build_uniform_landscape_grid_focus_story(clip_id)


def build_ch5_uniform_landscape_grid_d4_d2_zoom(clip_id):
    return _build_uniform_landscape_grid_d4_d2_zoom_story(clip_id)


def build_ch5_posterior_map_perturb(clip_id):
    return _build_posterior_map_perturb_story(clip_id)


def build_ch5_d4_origin_map_tutorial(clip_id):
    return _build_d4_origin_map_tutorial_story(clip_id)


def build_ch5_grid_2d_zoom_shadow_orbit(clip_id):
    return _build_grid_2d_zoom_shadow_orbit_story(clip_id, camera_pan=False)


def build_ch5_grid_2d_zoom_shadow_orbit_topview(clip_id):
    return _build_grid_2d_zoom_shadow_orbit_story(clip_id, camera_pan=True)


def build_ch5_grid_map_labeled_rotate90(clip_id):
    return _build_grid_map_labeled_rotate90_story(clip_id)


def build_ch5_belief_stem_surface(clip_id):
    return _build_belief_stem_surface_story(clip_id)


def build_ch5_d1_loglik_overlay(clip_id):
    return _build_d1_loglik_overlay_story(clip_id)


def build_ch5_uniform_landscape_squish_ct(clip_id):
    return _build_uniform_landscape_squish_ct_story(clip_id)


_CH5_LANDSCAPE_PARTS = ("squish", "ct")
_CH5_VOXEL_PRIORS = ("gaussian", "uniform")


def _ch5_build_export_specs() -> list[tuple[str, str, Callable[[str], list]]]:
    specs: list[tuple[str, str, Callable[[str], list]]] = []
    n = 1

    def add(slug: str, builder: Callable[[str], list]) -> None:
        nonlocal n
        cid = f"ch5_{n:02d}"
        specs.append((cid, f"{cid}_{slug}.mp4", builder))
        n += 1

    add("sequential_bars", build_ch5_02_sequential_bars)
    add("bars_to_density", build_ch5_03e)
    add("uniform_prior_sequential", build_ch5_03f)
    add("sequential_ct", build_ch5_03g)
    add("prior_sweep", build_ch5_04b)
    add("prior_bowl", build_ch5_04c)
    add("sequential", build_ch5_04d)
    add("decomposition", build_ch5_04e)
    for pk in CH5_DATASET_KEYS:
        add(f"{pk}_squish", _make_dataset_landscape_builder(pk, "squish", prior_kind="gaussian"))
        add(f"{pk}_squish_uniform", _make_dataset_landscape_builder(pk, "squish", prior_kind="uniform"))
        add(f"{pk}_ct", _make_dataset_landscape_builder(pk, "ct", prior_kind="gaussian"))
        add(f"{pk}_ct_uniform", _make_dataset_landscape_builder(pk, "ct", prior_kind="uniform"))
    add("montage_4x4", build_ch5_06_montage)
    add("duo_grid_2x2", build_ch5_07_duo_grid)
    add("duo_grid_2x2_uniform", build_ch5_07_duo_grid_uniform)
    add("ct_grid_2x2", build_ch5_07b_ct_grid)
    add("ct_grid_2x2_uniform", build_ch5_07b_ct_grid_uniform)
    add("credible_fill_D1", build_ch5_08a_credible_fill)
    add("voxel_axis_D1", build_ch5_08c_axis_D1)
    add("credible_4x4", build_ch5_10_credible_4x4)
    add("ct_D2", build_ch5_12b_ct_D2)
    for pk in CH5_DATASET_KEYS:
        for prior in _CH5_VOXEL_PRIORS:
            suffix = "" if prior == "gaussian" else "_uniform"
            add(
                f"voxel_{pk}{suffix}",
                (lambda k, p: (lambda clip_id: _build_credible_voxel_story(k, clip_id, prior_kind=p)))(pk, prior),
            )
    add("dataset_2d_map", build_ch5_dataset_2d_map)
    add("intro_to_01", build_ch5_intro_to_01)
    add("tutorial_to_prior", build_ch5_tutorial_to_prior)
    add("sequential_posterior_landscape", build_ch5_sequential_posterior_landscape)
    add("uniform_prior_to_sequential", build_ch5_uniform_prior_to_sequential)
    add("uniform_landscape_grid_2x2", build_ch5_uniform_landscape_grid)
    add("uniform_landscape_grid_zoom", build_ch5_uniform_landscape_grid_zoom)
    add("uniform_landscape_squish_ct", build_ch5_uniform_landscape_squish_ct)
    add("uniform_landscape_grid_focus", build_ch5_uniform_landscape_grid_focus)
    add("uniform_landscape_grid_d4_d2_zoom", build_ch5_uniform_landscape_grid_d4_d2_zoom)
    add("posterior_map_perturb", build_ch5_posterior_map_perturb)
    add("d4_origin_map_tutorial", build_ch5_d4_origin_map_tutorial)
    add("grid_2d_zoom_shadow_orbit", build_ch5_grid_2d_zoom_shadow_orbit)
    add("d1_loglik_overlay", build_ch5_d1_loglik_overlay)
    add("grid_2d_zoom_shadow_orbit_topview", build_ch5_grid_2d_zoom_shadow_orbit_topview)
    add("grid_map_labeled_rotate90", build_ch5_grid_map_labeled_rotate90)
    add("belief_stem_surface", build_ch5_belief_stem_surface)
    add("ct_to_voxel_grid_tour", build_ch5_ct_to_voxel_grid_tour)
    add("tutorial_to_prior_stem_view", build_ch5_tutorial_to_prior_stem_view)
    add("credible_interval_wander", build_ch5_credible_interval_wander)
    add("credible_thickness_wander", build_ch5_credible_thickness_wander)
    add("credible_axis_wander", build_ch5_credible_axis_wander)
    add("credible_thickness_wander_az0_45", build_ch5_credible_thickness_wander_az0_45)
    add("credible_axis_wander_D1_views", build_ch5_credible_axis_wander_D1_views)
    add("credible_map_basis_quiver_orbit", build_ch5_credible_map_basis_quiver_orbit)
    add("d3_belief_to_sigmoid_orbit", build_ch5_d3_belief_to_sigmoid_orbit)
    add("d1_fanout_credible", build_ch5_d1_fanout_credible)
    add("d2_noise_icon_emphasize", build_ch5_d2_noise_icon_emphasize)
    add("d1_best_line_howithink_d4_cuts", build_ch5_d1_best_line_howithink_d4_cuts)
    add("credibility_prior_intro", build_ch5_credibility_prior_intro)
    return specs


CH5_EXPORT_SPECS: list[tuple[str, str, Callable[[str], list]]] = _ch5_build_export_specs()


def ch5_export_clip(filename: str) -> Path:
    spec = next((s for s in CH5_EXPORT_SPECS if s[1] == filename or s[0] == filename), None)
    if spec is None:
        raise KeyError(f"unknown export: {filename!r}")
    clip_id, fn, builder = spec
    _ensure_packs()
    result = builder(clip_id)
    ms = int(
        CH5_STEM_SURF_MS if clip_id == "ch5_58"
        else CH5_HQ_LAND_MS if clip_id in (
            "ch5_29",
            "ch5_44", "ch5_45", "ch5_46", "ch5_47", "ch5_48", "ch5_49",
            "ch5_50", "ch5_51", "ch5_52", "ch5_53", "ch5_54", "ch5_55", "ch5_56",
            "ch5_57", "ch5_59", "ch5_60", "ch5_61", "ch5_62", "ch5_63",
            "ch5_64", "ch5_65", "ch5_66", "ch5_67", "ch5_68", "ch5_69", "ch5_70",
        )
        else CH5_PRIOR_LAND_MS if clip_id in ("ch5_06", "ch5_07")
        else CH5_MS
    )
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
    print("wrote", path, f"({n_frames} frames, text={_use_text(clip_id)}, ms={ms})")
    return path
