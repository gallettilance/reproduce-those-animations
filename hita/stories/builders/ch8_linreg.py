"""Chapter 8 — linear regression with plants: pack, plan, and render."""
from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from hita.config.profile import active_profile
from hita.export.spec import FrameSpec
from hita.primitives.linreg import (
    COND_MEANS,
    COND_SIGMA,
    HEIGHTS,
    X0_SLICE,
    ols_fit,
    plant_datasets,
    residuals,
    sample_conditional_ages,
    sample_shape_ages,
    shape_pdf,
    sse_ib,
    sse_mesh_ib,
    sse_two_point,
)
from hita.primitives.math import lerp, smoothstep
from hita.primitives.plants import icon_with_alpha, make_plant_icon, standing_plant_icon

# --- visual DNA (matches Ch1/Ch7 canvas + green pass colour) ---
FIGSIZE = (15.0, 9.5)
# Tight window for the linear story (clips 01–05); extras in 06–11 need the wide box.
XLIM = (2.0, 36.5)
YLIM = (0.0, 14.0)
XLIM_WIDE = (0.0, 40.0)
YLIM_WIDE = (0.0, 19.0)
PLANT_GREEN = "#2ca02c"
LINE_COLOR = "#1f4e79"
GUIDE_COLOR = "#9e9e9e"
MEAN_COLOR = "#c0392b"
HIST_COLOR = "#66bb6a"
BELL_COLOR = "#2e7d32"
SSE_COLOR = "#b71c1c"
TANGENT_COLOR = "#333333"
DOT_COLOR = "#1b5e20"

SSE_CMAP = LinearSegmentedColormap.from_list(
    "hita_sse_bowl",
    ["#fff8c0", "#aed581", "#43a047", "#1b5e20", "#0d3310"],
)

FONT_MULT = 2.0
ANNOT_MULT = 2.0
SURFACE_ALPHA_HI = 0.32
PLANT_ZOOM = 0.50
# Plant drawing occupies this vertical slice of the square icon (PIL y down).
ICON_TOP_FRAC = 0.18
ICON_BASE_FRAC = 0.88
ICON_PLANT_FRAC = ICON_BASE_FRAC - ICON_TOP_FRAC
ICON_BASE_ALIGN = 1.0 - ICON_BASE_FRAC  # AnnotationBbox y so the stem sits on the ground
POINT_S = 88.0
CLOUD_S = 54.0
LINE_LW = 3.8
GUIDE_LW = 2.2
SSE_LW = 4.2
BELL_LW = 3.2
SPINE_LW = 1.8
# Extra bottom/left margin so 2× axis labels are not clipped at the figure edge.
AX_BOTTOM = 0.175
AX_HEIGHT = 0.74
SCATTER_RECT = [0.12, AX_BOTTOM, 0.82, AX_HEIGHT]
# Equal duo panels: each half of the figure gets the same axes box; the
# right-hand ylabel ("Sum of Squared Errors") sits in the gutter, not on the left plot.
DUO_MARGIN = 0.12
DUO_W = 0.35
DUO_LEFT_X = DUO_MARGIN
DUO_RIGHT_X = 0.50 + DUO_MARGIN
WIDE_DATASETS = {"quad", "nquad", "hetero"}

# Clip 00: five plants on a baseline, one square-bracket dimension at a time.
# Wider than SCATTER_RECT (no axis labels) so 1 data-x equals 1 data-y in inches.
MEASURE_RECT = [0.05, AX_BOTTOM, 0.90, AX_HEIGHT]
_MEASURE_AX_ASPECT = (MEASURE_RECT[2] * FIGSIZE[0]) / (MEASURE_RECT[3] * FIGSIZE[1])
MEASURE_YLIM = (0.0, 8.8)
MEASURE_XLIM = (0.0, MEASURE_YLIM[1] * _MEASURE_AX_ASPECT)
MEASURE_GROUND = 0.90
MEASURE_MAX_H = 4.70
MEASURE_GAP = 0.42
MEASURE_LEFT = 1.55
BRACKET_PAD = 0.28
BRACKET_CAP = 0.34
BRACKET_H_SCALE = 1.13  # OffsetImage reads a bit taller than the data-height math
BRACKET_COLOR = "#333333"
BRACKET_LW = 3.6

P_GRID_LO, P_GRID_HI = 2.5, 11.5
# Bowl-centered window around plus OLS (intercept ≈ 1/3, slope ≈ 1/3).
# Tight enough that SSE fills the box as a cup, not a clipped plateau.
B0_LO, B0_HI = -0.4, 1.1
M_GRID_LO, M_GRID_HI = 0.10, 0.55
SSE2_XLIM = (5.0, 9.0)
SSE2_YLIM = (0.0, 38.0)  # 3D opening / full parabola
SSE2_YLIM_2D = (0.0, 12.0)  # clip-4 panel; SSE(5)=SSE(9)=10

BIN_EDGES = np.arange(0.0, 16.2, 0.2)  # 0.2-year bins across the full age axis
BAR_SCALE = 0.42
N_SAMP_HQ = 320
N_SAMP_DRAFT = 80
# Clip 02c: widen N(7, σ²) until the 3.2σ bell still sits inside YLIM.
SIGMA_WIDE = 2.10  # variance ≈ 4.4, mean stays 7

MARKER_START = (1.0, 0.12)  # inside the bowl, away from OLS


def _f(v: float) -> float:
    return round(float(v), 6)


def _fmt_num(v: float) -> str:
    return f"{v:.0f}" if abs(float(v) - round(float(v))) < 0.04 else f"{v:.1f}"


def _alphas(values) -> tuple[float, ...]:
    return tuple(_f(v) for v in values)


def _afs(base: float) -> float:
    """On-plot annotation size (2× the original)."""
    return float(base) * ANNOT_MULT


def _xy_lims(pack: dict[str, Any], spec: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    ds = str(spec.get("dataset", "core"))
    if ds in WIDE_DATASETS:
        xlim, ylim = tuple(pack["xlim_wide"]), tuple(pack["ylim_wide"])
    else:
        xlim, ylim = tuple(pack["xlim"]), tuple(pack["ylim"])
    if spec.get("xlim0") is not None:
        xlim = (float(spec["xlim0"]), float(xlim[1]))
    if spec.get("xlim1") is not None:
        xlim = (float(xlim[0]), float(spec["xlim1"]))
    return xlim, ylim


def _n_samples(ctx: dict[str, Any] | None = None) -> int:
    pack = (ctx or {}).get("_hita_pack") or {}
    if "n_samples" in pack:
        return int(pack["n_samples"])
    return N_SAMP_DRAFT if active_profile().is_draft() else N_SAMP_HQ


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------
def build_pack_ch8(clip_id: str, ctx: dict[str, Any]) -> dict[str, Any]:
    profile = active_profile()
    datasets = plant_datasets()
    n_mesh = 36 if profile.is_draft() else 72
    n_samp = N_SAMP_DRAFT if profile.is_draft() else N_SAMP_HQ
    p_grid = np.linspace(P_GRID_LO, P_GRID_HI, n_mesh)
    b0_grid = np.linspace(B0_LO, B0_HI, n_mesh)
    m_grid = np.linspace(M_GRID_LO, M_GRID_HI, n_mesh)

    plus = datasets["plus"]
    two_h = np.asarray([18.0, 18.0])
    two_a = np.asarray([6.0, 8.0])
    B0, M, z_full = sse_mesh_ib(plus.height, plus.age, b0_grid, m_grid)
    _, _, z_two = sse_mesh_ib(two_h, two_a, b0_grid, m_grid)

    fits: dict[str, tuple[float, float]] = {}
    resid: dict[str, np.ndarray] = {}
    for name, ds in datasets.items():
        a, b = ols_fit(ds.height, ds.age)
        fits[name] = (a, b)
        resid[name] = residuals(ds.height, ds.age, a, b)

    a_plus, b_plus = fits["plus"]
    p_ols = float(a_plus + b_plus * X0_SLICE)
    z_clip = float(np.ceil(float(np.max(z_full)) * 1.04 / 5.0) * 5.0)
    plant_lg, stem_align, plant_span_px = standing_plant_icon(256)

    samples = sample_conditional_ages(n=n_samp, seed=8, sigma=COND_SIGMA)
    # JSON/pickle-friendly keys
    samples_pack = {str(int(h)) if float(h).is_integer() else str(h): v for h, v in samples.items()}
    samples_shapes = {"normal": samples_pack}
    for kind in ("heavy", "skew", "bimodal"):
        raw = sample_shape_ages(n=n_samp, seed=8, kind=kind)
        samples_shapes[kind] = {
            str(int(h)) if float(h).is_integer() else str(h): v for h, v in raw.items()
        }
    bin_w = float(BIN_EDGES[1] - BIN_EDGES[0])
    peak_count = max(n_samp * 0.40 * bin_w, 1.0)
    bar_scale = 4.8 / peak_count

    rlim: dict[str, tuple[float, float]] = {}
    for name, r in resid.items():
        hi = float(max(3.5, 1.35 * np.max(np.abs(r)))) if r.size else 4.0
        rlim[name] = (-hi, hi)

    return {
        "clip_id": clip_id,
        "figsize": tuple(ctx.get("EXPORT_FIGSIZE", FIGSIZE)),
        "dpi": int(ctx.get("EXPORT_DPI", profile.export_dpi)),
        "font_size": float(ctx.get("FONT_SIZE", 13.75)) * FONT_MULT,
        "axis_label_size": float(ctx.get("AXIS_LABEL_SIZE", 15.0)) * FONT_MULT,
        "xlim": XLIM,
        "ylim": YLIM,
        "xlim_wide": XLIM_WIDE,
        "ylim_wide": YLIM_WIDE,
        "plant_icon": make_plant_icon(128),
        "plant_icon_lg": plant_lg,
        "plant_stem_align": float(stem_align),
        "plant_span_px": float(plant_span_px),
        "plant_icon_w": int(plant_lg.shape[1]),
        "datasets": {
            name: {"h": ds.height, "a": ds.age, "n_base": ds.n_base, "n": ds.n}
            for name, ds in datasets.items()
        },
        "fits": fits,
        "resid": resid,
        "resid_ylim": rlim,
        "samples": samples_pack,
        "samples_shapes": samples_shapes,
        "heights": np.asarray(HEIGHTS, dtype=np.float64),
        "cond_means": np.asarray(COND_MEANS, dtype=np.float64),
        "cond_sigma": float(COND_SIGMA),
        "sse_B0": B0,
        "sse_M": M,
        "sse_z_two": z_two,
        "sse_z_full": z_full,
        "sse_zlim": (0.0, z_clip),
        "p_ols": p_ols,
        "b0_ols": float(a_plus),
        "m_ols": float(b_plus),
        "p_grid": p_grid,
        "b0_grid": b0_grid,
        "m_grid": m_grid,
        "bin_edges": BIN_EDGES,
        "bar_scale": float(bar_scale),
        "n_samples": int(n_samp),
    }


# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------
class _Plan:
    def __init__(self, clip_id: str):
        self.clip_id = clip_id
        self.specs: list[FrameSpec] = []

    def add(self, kind: str, **params: Any) -> None:
        cleaned: dict[str, Any] = {"clip_id": self.clip_id, "kind": kind}
        for k, v in params.items():
            if isinstance(v, float):
                cleaned[k] = _f(v)
            elif isinstance(v, tuple) and v and isinstance(v[0], float):
                cleaned[k] = _alphas(v)
            else:
                cleaned[k] = v
        self.specs.append(FrameSpec(index=len(self.specs), kind=kind, params=cleaned))

    def hold(self, n: int, kind: str, **params: Any) -> None:
        for _ in range(max(0, int(n))):
            self.add(kind, **params)


def _fade_tuple(n: int, keep: set[int], u: float) -> tuple[float, ...]:
    """1 → 0 for indices not in keep; keep stays 1."""
    su = smoothstep(u)
    out = []
    for i in range(n):
        out.append(1.0 if i in keep else lerp(1.0, 0.0, su))
    return tuple(out)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
def plan_ch8_00(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Five plant icons of increasing height; one square bracket measures each."""
    p = active_profile()
    hold = p.scale_motion(10, 3)
    pop = p.scale_motion(14, 4)
    grow = p.scale_motion(18, 5)
    travel = p.scale_motion(16, 5)
    label_n = p.scale_motion(8, 3)
    n_h = len(HEIGHTS)
    plan = _Plan(clip_id)

    plan.hold(hold, "measure", n_shown=0, bracket_i=0, bracket_u=0.0, label_alpha=0.0)
    for i in range(n_h):
        for k in range(max(2, pop)):
            u = k / max(pop - 1, 1)
            plan.add(
                "measure",
                n_shown=i + 1,
                plant_pop=smoothstep(u),
                bracket_i=max(i - 1, 0),
                bracket_u=0.0 if i == 0 else 1.0,
                label_alpha=0.0 if i == 0 else 1.0,
            )
        if i == 0:
            for k in range(max(2, grow)):
                u = k / max(grow - 1, 1)
                su = smoothstep(u)
                plan.add(
                    "measure",
                    n_shown=1,
                    plant_pop=1.0,
                    bracket_i=0,
                    bracket_u=su,
                    label_alpha=smoothstep(max(0.0, (su - 0.55) / 0.45)),
                )
        else:
            for k in range(max(2, travel)):
                u = k / max(travel - 1, 1)
                plan.add(
                    "measure",
                    n_shown=i + 1,
                    plant_pop=1.0,
                    bracket_from=i - 1,
                    bracket_i=i,
                    bracket_u=smoothstep(u),
                    label_alpha=0.0,
                )
            for k in range(max(2, label_n)):
                u = k / max(label_n - 1, 1)
                plan.add(
                    "measure",
                    n_shown=i + 1,
                    plant_pop=1.0,
                    bracket_i=i,
                    bracket_u=1.0,
                    label_alpha=smoothstep(u),
                )
        plan.hold(
            hold,
            "measure",
            n_shown=i + 1,
            plant_pop=1.0,
            bracket_i=i,
            bracket_u=1.0,
            label_alpha=1.0,
        )
    plan.hold(
        hold,
        "measure",
        n_shown=n_h,
        plant_pop=1.0,
        bracket_i=n_h - 1,
        bracket_u=1.0,
        label_alpha=1.0,
    )
    return plan.specs


def _end_of_00() -> dict[str, Any]:
    n_h = len(HEIGHTS)
    return dict(
        n_shown=n_h,
        plant_pop=1.0,
        bracket_i=n_h - 1,
        bracket_u=1.0,
        label_alpha=1.0,
        scene_alpha=1.0,
    )


def plan_ch8_01(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    p = active_profile()
    hold = p.scale_motion(10, 3)
    fade_n = p.scale_motion(16, 5)
    axes_n = p.scale_motion(14, 4)
    pop = p.scale_motion(14, 4)
    guide = p.scale_motion(10, 3)
    line_n = p.scale_motion(24, 6)
    eq_n = p.scale_motion(14, 4)
    plan = _Plan(clip_id)
    end0 = _end_of_00()

    plan.hold(hold, "measure", **end0)
    for k in range(max(2, fade_n)):
        u = k / max(fade_n - 1, 1)
        su = smoothstep(u)
        plan.add(
            "measure",
            n_shown=end0["n_shown"],
            plant_pop=1.0,
            bracket_i=end0["bracket_i"],
            bracket_u=1.0,
            label_alpha=1.0 - su,
            scene_alpha=1.0 - su,
        )
    for k in range(max(2, axes_n)):
        u = k / max(axes_n - 1, 1)
        plan.add("scatter", dataset="core", n_shown=0, axes_alpha=smoothstep(u))
    plan.hold(hold, "scatter", dataset="core", n_shown=0, axes_alpha=1.0)
    for i in range(5):
        for k in range(max(2, pop)):
            u = k / max(pop - 1, 1)
            plan.add(
                "scatter",
                dataset="core",
                n_shown=i + 1,
                guide_index=i,
                guide_u=smoothstep(min(1.0, u * 1.15)),
                plant_pop=smoothstep(u),
            )
        plan.hold(
            guide,
            "scatter",
            dataset="core",
            n_shown=i + 1,
            guide_index=i,
            guide_u=1.0,
            plant_pop=1.0,
        )
    plan.hold(max(2, hold // 2), "scatter", dataset="core", n_shown=5, plant_pop=1.0)
    for k in range(max(2, line_n)):
        u = k / max(line_n - 1, 1)
        plan.add("scatter", dataset="core", n_shown=5, line_u=smoothstep(u), plant_pop=1.0)
    for k in range(max(2, eq_n)):
        u = k / max(eq_n - 1, 1)
        plan.add(
            "scatter",
            dataset="core",
            n_shown=5,
            line_u=1.0,
            eq_alpha=smoothstep(u),
            plant_pop=1.0,
        )
    plan.hold(
        hold,
        "scatter",
        dataset="core",
        n_shown=5,
        line_u=1.0,
        eq_alpha=1.0,
        plant_pop=1.0,
    )
    return plan.specs


def _end_of_01() -> dict[str, Any]:
    return dict(dataset="core", n_shown=5, line_u=1.0, eq_alpha=1.0, plant_pop=1.0)


def _end_quiet() -> dict[str, Any]:
    return dict(dataset="core", n_shown=5, line_u=1.0, eq_alpha=0.0, plant_pop=1.0)


def plan_ch8_01b(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Rise/run triangle, then open the x-axis to show intercept = 0."""
    p = active_profile()
    hold = p.scale_motion(10, 3)
    tri_n = p.scale_motion(22, 6)
    lab_n = p.scale_motion(12, 4)
    open_n = p.scale_motion(20, 5)
    plan = _Plan(clip_id)
    end1 = _end_of_01()
    plan.hold(hold, "scatter", **end1)
    for k in range(max(2, tri_n)):
        u = k / max(tri_n - 1, 1)
        plan.add("scatter", slope_tri=smoothstep(u), **end1)
    for k in range(max(2, lab_n)):
        u = k / max(lab_n - 1, 1)
        plan.add("scatter", slope_tri=1.0, slope_lab=smoothstep(u), **end1)
    plan.hold(hold, "scatter", slope_tri=1.0, slope_lab=1.0, **end1)
    for k in range(max(2, open_n)):
        u = k / max(open_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "scatter",
            slope_tri=1.0,
            slope_lab=1.0,
            xlim0=lerp(2.0, -0.6, s),
            intercept_u=s,
            **end1,
        )
    plan.hold(
        hold,
        "scatter",
        slope_tri=1.0,
        slope_lab=1.0,
        xlim0=-0.6,
        intercept_u=1.0,
        **end1,
    )
    return plan.specs


def plan_ch8_01c(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """The OLS line always passes through the centroid (x̄, ȳ)."""
    p = active_profile()
    hold = p.scale_motion(16, 5)
    mark_n = p.scale_motion(24, 6)
    plan = _Plan(clip_id)
    end1 = _end_of_01()
    plan.hold(hold, "scatter", **end1)
    for k in range(max(2, mark_n)):
        u = k / max(mark_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "scatter",
            centroid_u=s,
            eq_alpha=lerp(1.0, 0.0, s),
            story_caption=r"the line always goes through $(\bar{x},\bar{y})$",
            story_alpha=s,
            dataset="core",
            n_shown=5,
            line_u=1.0,
            plant_pop=1.0,
        )
    plan.hold(
        hold,
        "scatter",
        centroid_u=1.0,
        eq_alpha=0.0,
        story_caption=r"the line always goes through $(\bar{x},\bar{y})$",
        story_alpha=1.0,
        dataset="core",
        n_shown=5,
        line_u=1.0,
        plant_pop=1.0,
    )
    return plan.specs


def plan_ch8_01d(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Tilt the perfect line down to the average, then back — residuals appear and vanish."""
    p = active_profile()
    hold = p.scale_motion(10, 3)
    morph_n = p.scale_motion(28, 7)
    plan = _Plan(clip_id)
    end1 = _end_of_01()
    xbar, ybar = 18.0, 6.0
    plan.hold(hold, "scatter", **end1)

    def _through_mean(m: float) -> tuple[float, float]:
        return ybar - m * xbar, m

    for k in range(max(2, morph_n)):
        u = k / max(morph_n - 1, 1)
        s = smoothstep(u)
        b0, b1 = _through_mean(lerp(1.0 / 3.0, 0.0, s))
        plan.add(
            "scatter",
            dataset="core",
            n_shown=5,
            plant_pop=1.0,
            line_u=1.0,
            b0=b0,
            b1=b1,
            eq_alpha=lerp(1.0, 0.0, s),
            resid_sticks=s,
            sse_badge=s,
            story_caption="just the average",
            story_alpha=s,
        )
    b0_m, b1_m = _through_mean(0.0)
    plan.hold(
        hold,
        "scatter",
        dataset="core",
        n_shown=5,
        plant_pop=1.0,
        line_u=1.0,
        b0=b0_m,
        b1=b1_m,
        resid_sticks=1.0,
        sse_badge=1.0,
        story_caption="just the average",
        story_alpha=1.0,
    )
    for k in range(max(2, morph_n)):
        u = k / max(morph_n - 1, 1)
        s = smoothstep(u)
        b0, b1 = _through_mean(lerp(0.0, 1.0 / 3.0, s))
        plan.add(
            "scatter",
            dataset="core",
            n_shown=5,
            plant_pop=1.0,
            line_u=1.0,
            b0=b0,
            b1=b1,
            eq_alpha=s,
            resid_sticks=1.0 - s,
            sse_badge=1.0,
            story_caption="a line fits perfectly",
            story_alpha=1.0,
        )
    plan.hold(hold, "scatter", sse_badge=1.0, story_caption="a line fits perfectly", story_alpha=1.0, **end1)
    return plan.specs


def plan_ch8_01e(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Predict age at a new height, then extrapolate past the data."""
    p = active_profile()
    hold = p.scale_motion(10, 3)
    pred_n = p.scale_motion(22, 6)
    plan = _Plan(clip_id)
    end1 = _end_of_01()
    plan.hold(hold, "scatter", **end1)
    for k in range(max(2, pred_n)):
        u = k / max(pred_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "scatter",
            pred_x=15.0,
            pred_u=s,
            eq_alpha=lerp(1.0, 0.0, s),
            story_caption="a new 15 cm plant",
            story_alpha=s,
            dataset="core",
            n_shown=5,
            line_u=1.0,
            plant_pop=1.0,
        )
    plan.hold(
        hold,
        "scatter",
        pred_x=15.0,
        pred_u=1.0,
        eq_alpha=0.0,
        story_caption="a new 15 cm plant",
        story_alpha=1.0,
        dataset="core",
        n_shown=5,
        line_u=1.0,
        plant_pop=1.0,
    )
    for k in range(max(2, pred_n)):
        u = k / max(pred_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "scatter",
            extra_h=15.0,
            extra_a=5.0,
            extra_alpha=1.0,
            pred_x=33.0,
            pred_u=s,
            xlim1=lerp(36.5, 38.0, s),
            eq_alpha=0.0,
            story_caption="beyond the data  —  extrapolation",
            story_alpha=s,
            dataset="core",
            n_shown=5,
            line_u=1.0,
            plant_pop=1.0,
        )
    plan.hold(
        hold,
        "scatter",
        extra_h=15.0,
        extra_a=5.0,
        extra_alpha=1.0,
        pred_x=33.0,
        pred_u=1.0,
        xlim1=38.0,
        eq_alpha=0.0,
        story_caption="beyond the data  —  extrapolation",
        story_alpha=1.0,
        dataset="core",
        n_shown=5,
        line_u=1.0,
        plant_pop=1.0,
    )
    return plan.specs


def plan_ch8_01f(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """A sixth plant misses the line — ŷ on the line, residual up to the plant."""
    p = active_profile()
    hold = p.scale_motion(10, 3)
    pop_n = p.scale_motion(16, 5)
    stick_n = p.scale_motion(16, 5)
    mark_n = p.scale_motion(14, 4)
    plan = _Plan(clip_id)
    end1 = _end_of_01()
    plan.hold(hold, "scatter", **end1)
    plus = dict(dataset="plus", n_shown=6, line_u=1.0, plant_pop=1.0, eq_alpha=0.0)
    for k in range(max(2, pop_n)):
        u = k / max(pop_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "scatter",
            alphas=_alphas((1.0, 1.0, 1.0, 1.0, 1.0, s)),
            story_caption="a sixth plant misses the line",
            story_alpha=s,
            **plus,
        )
    plan.hold(
        hold,
        "scatter",
        alphas=_alphas((1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
        story_caption="a sixth plant misses the line",
        story_alpha=1.0,
        **plus,
    )
    for k in range(max(2, stick_n)):
        u = k / max(stick_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "scatter",
            alphas=_alphas((1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
            resid_sticks=s,
            story_caption=r"the leftover is the residual  $r = y - \hat{y}$",
            story_alpha=s,
            **plus,
        )
    plan.hold(
        hold,
        "scatter",
        alphas=_alphas((1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
        resid_sticks=1.0,
        story_caption=r"the leftover is the residual  $r = y - \hat{y}$",
        story_alpha=1.0,
        **plus,
    )
    for k in range(max(2, mark_n)):
        u = k / max(mark_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "scatter",
            alphas=_alphas((1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
            resid_sticks=1.0,
            yhat_marks=s,
            story_caption=r"$\hat{y}$ is the line's guess for each plant",
            story_alpha=s,
            **plus,
        )
    plan.hold(
        hold,
        "scatter",
        alphas=_alphas((1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
        resid_sticks=1.0,
        yhat_marks=1.0,
        story_caption=r"$\hat{y}$ is the line's guess for each plant",
        story_alpha=1.0,
        **plus,
    )
    return plan.specs


def plan_ch8_01g(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Total squares around the average vs leftover squares around the line — R²."""
    p = active_profile()
    hold = p.scale_motion(10, 3)
    sq_n = p.scale_motion(16, 5)
    morph_n = p.scale_motion(32, 8)
    plan = _Plan(clip_id)
    ybar = 38.0 / 6.0
    plus = dict(dataset="plus", n_shown=6, plant_pop=1.0, line_u=1.0, eq_alpha=0.0)
    plan.hold(hold, "scatter", resid_sticks=1.0, **plus)

    def _line_at(s_mean: float) -> tuple[float, float]:
        return lerp(0.0, ybar, s_mean), lerp(1.0 / 3.0, 0.0, s_mean)

    for k in range(max(2, sq_n)):
        u = k / max(sq_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "scatter",
            resid_sticks=1.0,
            resid_squares=s,
            r2_badge=s,
            story_caption="leftover variation around the line",
            story_alpha=s,
            **plus,
        )
    plan.hold(
        hold,
        "scatter",
        resid_sticks=1.0,
        resid_squares=1.0,
        r2_badge=1.0,
        story_caption="leftover variation around the line",
        story_alpha=1.0,
        **plus,
    )
    for k in range(max(2, morph_n)):
        u = k / max(morph_n - 1, 1)
        s = smoothstep(u)
        b0, b1 = _line_at(s)
        plan.add(
            "scatter",
            b0=b0,
            b1=b1,
            resid_sticks=1.0,
            resid_squares=1.0,
            r2_badge=1.0,
            story_caption="total variation around the average",
            story_alpha=s,
            **plus,
        )
    b0_m, b1_m = _line_at(1.0)
    plan.hold(
        hold,
        "scatter",
        b0=b0_m,
        b1=b1_m,
        resid_sticks=1.0,
        resid_squares=1.0,
        r2_badge=1.0,
        story_caption="total variation around the average",
        story_alpha=1.0,
        **plus,
    )
    for k in range(max(2, morph_n)):
        u = k / max(morph_n - 1, 1)
        s = smoothstep(u)
        b0, b1 = _line_at(1.0 - s)
        plan.add(
            "scatter",
            b0=b0,
            b1=b1,
            resid_sticks=1.0,
            resid_squares=1.0,
            r2_badge=1.0,
            story_caption=r"the line explains most of it",
            story_alpha=s,
            **plus,
        )
    plan.hold(
        hold,
        "scatter",
        resid_sticks=1.0,
        resid_squares=1.0,
        r2_badge=1.0,
        story_caption=r"the line explains most of it",
        story_alpha=1.0,
        **plus,
    )
    return plan.specs


def _plan_02_to_two_points(plan: _Plan, ctx: dict[str, Any]) -> dict[str, Any]:
    """Shared 02 prefix: end of clip 1, pop (18, 8), fade everyone else."""
    p = active_profile()
    hold = p.scale_motion(8, 2)
    pop = p.scale_motion(14, 4)
    fade_n = p.scale_motion(18, 5)
    end1 = dict(dataset="plus", n_shown=5, line_u=1.0, eq_alpha=1.0, plant_pop=1.0)
    plan.hold(hold, "scatter", **end1)
    for k in range(max(2, pop)):
        u = k / max(pop - 1, 1)
        plan.add(
            "scatter",
            dataset="plus",
            n_shown=6,
            line_u=1.0,
            eq_alpha=1.0,
            guide_index=5,
            guide_u=smoothstep(min(1.0, u * 1.15)),
            plant_pop=smoothstep(u),
        )
    keep = {2, 5}
    for k in range(max(2, fade_n)):
        u = k / max(fade_n - 1, 1)
        plan.add(
            "scatter",
            dataset="plus",
            n_shown=6,
            line_u=lerp(1.0, 0.0, smoothstep(u)),
            eq_alpha=lerp(1.0, 0.0, smoothstep(u)),
            alphas=_fade_tuple(6, keep, u),
            plant_pop=1.0,
        )
    return dict(
        dataset="plus",
        n_shown=6,
        alphas=_alphas(_fade_tuple(6, keep, 1.0)),
        plant_pop=1.0,
    )


def plan_ch8_02(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    p = active_profile()
    hold = p.scale_motion(8, 2)
    samp_n = p.scale_motion(56, 12)
    bell_n = p.scale_motion(20, 5)
    mean_n = p.scale_motion(12, 4)
    plan = _Plan(clip_id)
    faded = _plan_02_to_two_points(plan, ctx)
    n_samp = _n_samples(ctx)
    hist18 = (0, 0, 0, 0, 0)
    for k in range(max(2, samp_n)):
        u = k / max(samp_n - 1, 1)
        n_show = int(round(u * n_samp))
        hn = (0, 0, n_show, 0, 0)
        plan.add("scatter", hist_n=hn, hist_bar_u=1.0, **faded)
        hist18 = hn
    for k in range(max(2, bell_n)):
        u = k / max(bell_n - 1, 1)
        ba = (0.0, 0.0, smoothstep(u), 0.0, 0.0)
        plan.add("scatter", hist_n=hist18, hist_bar_u=1.0, bell_alpha=ba, **faded)
    ba = (0.0, 0.0, 1.0, 0.0, 0.0)
    for k in range(max(2, mean_n)):
        u = k / max(mean_n - 1, 1)
        ma = (0.0, 0.0, smoothstep(u), 0.0, 0.0)
        plan.add(
            "scatter",
            hist_n=hist18,
            hist_bar_u=1.0,
            bell_alpha=ba,
            mean_alpha=ma,
            **faded,
        )
    plan.hold(
        hold,
        "scatter",
        hist_n=hist18,
        hist_bar_u=1.0,
        bell_alpha=ba,
        mean_alpha=(0.0, 0.0, 1.0, 0.0, 0.0),
        **faded,
    )
    return plan.specs


def plan_ch8_02b(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Same setup as 02, but samples arrive as points first, then the histogram grows."""
    p = active_profile()
    hold = p.scale_motion(8, 2)
    cloud_n_frames = p.scale_motion(64, 14)
    hist_n_frames = p.scale_motion(28, 7)
    bell_n = p.scale_motion(20, 5)
    mean_n = p.scale_motion(12, 4)
    plan = _Plan(clip_id)
    faded = _plan_02_to_two_points(plan, ctx)
    n_samp = _n_samples(ctx)
    for k in range(max(2, cloud_n_frames)):
        u = k / max(cloud_n_frames - 1, 1)
        n_show = int(round(smoothstep(u) * n_samp))
        plan.add("scatter", cloud_n=n_show, cloud_alpha=1.0, cloud_h=2, **faded)
    cloud_full = dict(cloud_n=n_samp, cloud_h=2, **faded)
    plan.hold(max(2, hold // 2), "scatter", cloud_alpha=1.0, **cloud_full)
    hist18 = (0, 0, n_samp, 0, 0)
    for k in range(max(2, hist_n_frames)):
        u = k / max(hist_n_frames - 1, 1)
        s = smoothstep(u)
        plan.add(
            "scatter",
            hist_n=hist18,
            hist_bar_u=s,
            cloud_alpha=lerp(1.0, 0.22, s),
            **cloud_full,
        )
    for k in range(max(2, bell_n)):
        u = k / max(bell_n - 1, 1)
        ba = (0.0, 0.0, smoothstep(u), 0.0, 0.0)
        plan.add(
            "scatter",
            hist_n=hist18,
            hist_bar_u=1.0,
            bell_alpha=ba,
            cloud_alpha=0.22,
            **cloud_full,
        )
    ba = (0.0, 0.0, 1.0, 0.0, 0.0)
    for k in range(max(2, mean_n)):
        u = k / max(mean_n - 1, 1)
        ma = (0.0, 0.0, smoothstep(u), 0.0, 0.0)
        plan.add(
            "scatter",
            hist_n=hist18,
            hist_bar_u=1.0,
            bell_alpha=ba,
            mean_alpha=ma,
            cloud_alpha=0.22,
            **cloud_full,
        )
    plan.hold(
        hold,
        "scatter",
        hist_n=hist18,
        hist_bar_u=1.0,
        bell_alpha=ba,
        mean_alpha=(0.0, 0.0, 1.0, 0.0, 0.0),
        cloud_alpha=0.22,
        **cloud_full,
    )
    return plan.specs


def _end_of_02(ctx: dict[str, Any]) -> dict[str, Any]:
    n_samp = _n_samples(ctx)
    keep = {2, 5}
    return dict(
        dataset="plus",
        n_shown=6,
        alphas=_alphas(_fade_tuple(6, keep, 1.0)),
        plant_pop=1.0,
        hist_n=(0, 0, n_samp, 0, 0),
        hist_bar_u=1.0,
        bell_alpha=(0.0, 0.0, 1.0, 0.0, 0.0),
        mean_alpha=(0.0, 0.0, 1.0, 0.0, 0.0),
    )


def plan_ch8_02c(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """From the end of clip 02, spread Age | Height=18 around 7 (σ: 1 → 2.1)."""
    p = active_profile()
    hold = p.scale_motion(10, 3)
    spread_n = p.scale_motion(48, 10)
    plan = _Plan(clip_id)
    end2 = _end_of_02(ctx)
    plan.hold(hold, "scatter", sigma_mult=1.0, **end2)
    for k in range(max(2, spread_n)):
        u = k / max(spread_n - 1, 1)
        plan.add(
            "scatter",
            sigma_mult=lerp(1.0, SIGMA_WIDE, smoothstep(u)),
            **end2,
        )
    plan.hold(hold, "scatter", sigma_mult=SIGMA_WIDE, **end2)
    return plan.specs


def plan_ch8_03(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    p = active_profile()
    hold = p.scale_motion(8, 2)
    samp_n = p.scale_motion(36, 8)
    bell_n = p.scale_motion(14, 4)
    line_n = p.scale_motion(20, 5)
    plan = _Plan(clip_id)
    n_samp = _n_samples(ctx)
    keep = {2, 5}
    faded = dict(
        dataset="plus",
        n_shown=6,
        alphas=_alphas(_fade_tuple(6, keep, 1.0)),
        plant_pop=1.0,
        hist_bar_u=1.0,
        bell_alpha=(0.0, 0.0, 1.0, 0.0, 0.0),
        mean_alpha=(0.0, 0.0, 1.0, 0.0, 0.0),
        hist_n=(0, 0, n_samp, 0, 0),
    )
    plan.hold(hold, "scatter", **faded)

    others = (0, 1, 3, 4)
    for k in range(max(2, samp_n)):
        u = k / max(samp_n - 1, 1)
        n_show = int(round(u * n_samp))
        hn = [0, 0, n_samp, 0, 0]
        al = list(_fade_tuple(6, keep, 1.0))
        for j in others:
            hn[j] = n_show
            # bring original plants at those heights back as samples arrive
            orig_idx = j if j < 5 else 5
            al[orig_idx] = smoothstep(u)
        plan.add(
            "scatter",
            dataset="plus",
            n_shown=6,
            alphas=tuple(al),
            plant_pop=1.0,
            hist_n=tuple(hn),
            hist_bar_u=1.0,
            bell_alpha=(0.0, 0.0, 1.0, 0.0, 0.0),
            mean_alpha=(0.0, 0.0, 1.0, 0.0, 0.0),
        )
    al_full = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    hn_full = (n_samp,) * 5
    for k in range(max(2, bell_n)):
        u = k / max(bell_n - 1, 1)
        s = smoothstep(u)
        ba = (s, s, 1.0, s, s)
        ma = (s, s, 1.0, s, s)
        plan.add(
            "scatter",
            dataset="plus",
            n_shown=6,
            alphas=al_full,
            plant_pop=1.0,
            hist_n=hn_full,
            hist_bar_u=1.0,
            bell_alpha=ba,
            mean_alpha=ma,
        )
    ba = (1.0, 1.0, 1.0, 1.0, 1.0)
    ma = ba
    for k in range(max(2, line_n)):
        u = k / max(line_n - 1, 1)
        plan.add(
            "scatter",
            dataset="plus",
            n_shown=6,
            alphas=al_full,
            plant_pop=1.0,
            hist_n=hn_full,
            hist_bar_u=1.0,
            bell_alpha=ba,
            mean_alpha=ma,
            line_u=smoothstep(u),
            eq_alpha=smoothstep(u),
        )
    plan.hold(
        hold,
        "scatter",
        dataset="plus",
        n_shown=6,
        alphas=al_full,
        plant_pop=1.0,
        hist_n=hn_full,
        hist_bar_u=1.0,
        bell_alpha=ba,
        mean_alpha=ma,
        line_u=1.0,
        eq_alpha=1.0,
    )
    return plan.specs


def _end_of_03(ctx: dict[str, Any]) -> dict[str, Any]:
    n_samp = _n_samples(ctx)
    return dict(
        dataset="plus",
        n_shown=6,
        alphas=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        plant_pop=1.0,
        hist_n=(n_samp,) * 5,
        hist_bar_u=1.0,
        bell_alpha=(1.0, 1.0, 1.0, 1.0, 1.0),
        mean_alpha=(1.0, 1.0, 1.0, 1.0, 1.0),
        line_u=1.0,
        eq_alpha=1.0,
    )


_SHAPE_CAPTION = {
    "heavy": "long tails",
    "skew": "same-side tails",
    "bimodal": "bimodal",
}


def plan_ch8_03b(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """From the end of clip 03, morph Age|Height shapes with the means held fixed."""
    p = active_profile()
    hold = p.scale_motion(10, 3)
    morph_n = p.scale_motion(40, 8)
    plan = _Plan(clip_id)
    base = _end_of_03(ctx)
    plan.hold(hold, "scatter", **base)

    def morph(kind_a: str, kind_b: str, caption: str) -> None:
        for k in range(max(2, morph_n)):
            u = k / max(morph_n - 1, 1)
            s = smoothstep(u)
            cap = caption if kind_b != "normal" else _SHAPE_CAPTION.get(kind_a, "")
            cap_a = s if kind_b != "normal" else (1.0 - s)
            plan.add(
                "scatter",
                shape_a=kind_a,
                shape_b=kind_b,
                shape_u=s,
                dist_caption=cap,
                dist_caption_alpha=cap_a,
                **base,
            )
        hold_kw = dict(base)
        if kind_b != "normal":
            hold_kw.update(
                shape_a=kind_b,
                shape_b=kind_b,
                shape_u=1.0,
                dist_caption=caption,
                dist_caption_alpha=1.0,
            )
        plan.hold(hold, "scatter", **hold_kw)

    morph("normal", "heavy", "long tails")
    morph("heavy", "skew", "same-side tails")
    morph("skew", "bimodal", "bimodal")
    morph("bimodal", "normal", "")
    return plan.specs


def plan_ch8_04a(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Two height-18 plants: residual sticks become squares whose areas are SSE."""
    p = active_profile()
    hold = p.scale_motion(8, 2)
    fade_n = p.scale_motion(18, 5)
    stick_n = p.scale_motion(14, 4)
    sq_n = p.scale_motion(18, 5)
    sweep_n = p.scale_motion(48, 10)
    to7_n = p.scale_motion(24, 6)
    plan = _Plan(clip_id)
    keep = {2, 5}
    start = dict(dataset="plus", n_shown=6, plant_pop=1.0, line_u=1.0, eq_alpha=1.0)
    plan.hold(hold, "scatter", alphas=(1.0,) * 6, **start)
    for k in range(max(2, fade_n)):
        u = k / max(fade_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "scatter",
            alphas=_fade_tuple(6, keep, u),
            line_u=lerp(1.0, 0.0, s),
            eq_alpha=lerp(1.0, 0.0, s),
            dataset="plus",
            n_shown=6,
            plant_pop=1.0,
        )
    two = dict(
        dataset="plus",
        n_shown=6,
        alphas=_alphas(_fade_tuple(6, keep, 1.0)),
        plant_pop=1.0,
        pred_p=6.0,
        pred_line_alpha=1.0,
    )
    plan.hold(max(2, hold // 2), "scatter", **two)
    for k in range(max(2, stick_n)):
        u = k / max(stick_n - 1, 1)
        plan.add("scatter", resid_sticks=smoothstep(u), **two)
    for k in range(max(2, sq_n)):
        u = k / max(sq_n - 1, 1)
        plan.add(
            "scatter",
            resid_sticks=1.0,
            resid_squares=smoothstep(u),
            resid_lab=smoothstep(u),
            **two,
        )
    plan.hold(hold, "scatter", resid_sticks=1.0, resid_squares=1.0, resid_lab=1.0, **two)
    for k in range(max(2, sweep_n)):
        u = k / max(sweep_n - 1, 1)
        p_now = lerp(6.0, 8.5, smoothstep(u))
        plan.add(
            "scatter",
            resid_sticks=1.0,
            resid_squares=1.0,
            resid_lab=1.0,
            dataset="plus",
            n_shown=6,
            alphas=_alphas(_fade_tuple(6, keep, 1.0)),
            plant_pop=1.0,
            pred_p=p_now,
            pred_line_alpha=1.0,
        )
    for k in range(max(2, to7_n)):
        u = k / max(to7_n - 1, 1)
        p_now = lerp(8.5, 7.0, smoothstep(u))
        plan.add(
            "scatter",
            resid_sticks=1.0,
            resid_squares=1.0,
            resid_lab=1.0,
            dataset="plus",
            n_shown=6,
            alphas=_alphas(_fade_tuple(6, keep, 1.0)),
            plant_pop=1.0,
            pred_p=p_now,
            pred_line_alpha=1.0,
            story_caption="smallest squares at $p=7$",
            story_alpha=smoothstep(u),
        )
    plan.hold(
        hold,
        "scatter",
        resid_sticks=1.0,
        resid_squares=1.0,
        resid_lab=1.0,
        dataset="plus",
        n_shown=6,
        alphas=_alphas(_fade_tuple(6, keep, 1.0)),
        plant_pop=1.0,
        pred_p=7.0,
        pred_line_alpha=1.0,
        story_caption="smallest squares at $p=7$",
        story_alpha=1.0,
    )
    return plan.specs


def plan_ch8_04b(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Absolute errors are tied anywhere in [6, 8]; squares uniquely pick p=7."""
    p = active_profile()
    hold = p.scale_motion(10, 3)
    fade_n = p.scale_motion(16, 5)
    sweep_n = p.scale_motion(40, 10)
    plan = _Plan(clip_id)
    keep = {2, 5}
    two = dict(
        dataset="plus",
        n_shown=6,
        alphas=_alphas(_fade_tuple(6, keep, 1.0)),
        plant_pop=1.0,
        pred_line_alpha=1.0,
        eq_alpha=0.0,
    )
    plan.hold(
        hold,
        "scatter",
        resid_sticks=1.0,
        resid_squares=1.0,
        resid_lab=1.0,
        pred_p=7.0,
        story_caption="squares pick one $p$",
        story_alpha=1.0,
        **two,
    )
    for k in range(max(2, fade_n)):
        u = k / max(fade_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "scatter",
            resid_sticks=1.0,
            resid_squares=1.0 - s,
            resid_lab=1.0,
            l1_lab=s,
            pred_p=7.0,
            story_caption=r"absolute errors: any $p$ between 6 and 8",
            story_alpha=s,
            **two,
        )
    plan.hold(
        hold,
        "scatter",
        resid_sticks=1.0,
        resid_lab=1.0,
        l1_lab=1.0,
        pred_p=7.0,
        story_caption=r"absolute errors: any $p$ between 6 and 8",
        story_alpha=1.0,
        **two,
    )
    p_path = [7.0, 6.0, 8.0, 5.4, 7.0]
    for a, b in zip(p_path, p_path[1:]):
        n = sweep_n if abs(b - a) > 1.2 else max(sweep_n // 2, 8)
        for k in range(max(2, n)):
            u = k / max(n - 1, 1)
            p_now = lerp(a, b, smoothstep(u))
            inside = 6.0 - 0.04 <= p_now <= 8.0 + 0.04
            cap = (
                r"absolute errors: any $p$ between 6 and 8"
                if inside
                else "outside $[6,8]$ the tie breaks"
            )
            plan.add(
                "scatter",
                resid_sticks=1.0,
                resid_lab=1.0,
                l1_lab=1.0,
                pred_p=p_now,
                story_caption=cap,
                story_alpha=1.0,
                **two,
            )
    for k in range(max(2, fade_n)):
        u = k / max(fade_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "scatter",
            resid_sticks=1.0,
            resid_squares=s,
            resid_lab=1.0,
            l1_lab=1.0 - s,
            pred_p=7.0,
            story_caption="that's why we square",
            story_alpha=s,
            **two,
        )
    plan.hold(
        hold,
        "scatter",
        resid_sticks=1.0,
        resid_squares=1.0,
        resid_lab=1.0,
        pred_p=7.0,
        story_caption="that's why we square",
        story_alpha=1.0,
        **two,
    )
    return plan.specs


def plan_ch8_04(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    p = active_profile()
    hold = p.scale_motion(8, 2)
    fade_n = p.scale_motion(18, 5)
    split_n = p.scale_motion(26, 6)
    sweep_n = p.scale_motion(52, 12)
    to6_n = p.scale_motion(22, 5)
    to7_n = p.scale_motion(28, 6)
    min_n = p.scale_motion(14, 4)
    n_samp = _n_samples(ctx)
    plan = _Plan(clip_id)
    al_full = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    start = dict(
        dataset="plus",
        n_shown=6,
        alphas=al_full,
        plant_pop=1.0,
        hist_n=(n_samp,) * 5,
        hist_bar_u=1.0,
        bell_alpha=(1.0,) * 5,
        mean_alpha=(1.0,) * 5,
        line_u=1.0,
        eq_alpha=1.0,
    )
    plan.hold(hold, "scatter", **start)
    keep = {2, 5}
    for k in range(max(2, fade_n)):
        u = k / max(fade_n - 1, 1)
        s = smoothstep(u)
        z = lerp(1.0, 0.0, s)
        plan.add(
            "scatter",
            dataset="plus",
            n_shown=6,
            alphas=_fade_tuple(6, keep, u),
            plant_pop=1.0,
            hist_n=(n_samp,) * 5,
            hist_bar_u=z,
            bell_alpha=(z,) * 5,
            mean_alpha=(z,) * 5,
            line_u=z,
            eq_alpha=z,
        )
    two = dict(
        dataset="plus",
        n_shown=6,
        alphas=_alphas(_fade_tuple(6, keep, 1.0)),
        plant_pop=1.0,
    )
    plan.hold(max(2, hold // 2), "scatter", **two)
    for k in range(max(2, split_n)):
        u = k / max(split_n - 1, 1)
        plan.add("duo_sse", split_u=smoothstep(u), **two)
    p_lo, p_hi = SSE2_XLIM
    # Slide the left-plot prediction through the p range, tracing SSE on the right.
    for k in range(max(2, sweep_n)):
        u = k / max(sweep_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "duo_sse",
            split_u=1.0,
            pred_p=lerp(p_lo, p_hi, s),
            sse_trail=s,
            pred_line_alpha=1.0,
            **two,
        )
    plan.hold(
        max(2, hold // 2),
        "duo_sse",
        split_u=1.0,
        pred_p=p_hi,
        sse_trail=1.0,
        pred_line_alpha=1.0,
        **two,
    )
    # Pick p = 6 and show its SSE with a dotted guide.
    for k in range(max(2, to6_n)):
        u = k / max(to6_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "duo_sse",
            split_u=1.0,
            pred_p=lerp(p_hi, 6.0, s),
            sse_trail=1.0,
            pred_line_alpha=1.0,
            sse_guide_alpha=s,
            **two,
        )
    plan.hold(
        hold,
        "duo_sse",
        split_u=1.0,
        pred_p=6.0,
        sse_trail=1.0,
        pred_line_alpha=1.0,
        sse_guide_alpha=1.0,
        **two,
    )
    # Slide p: 6 → 7; the probe and SSE guide drop to the minimum.
    for k in range(max(2, to7_n)):
        u = k / max(to7_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "duo_sse",
            split_u=1.0,
            pred_p=lerp(6.0, 7.0, s),
            sse_trail=1.0,
            pred_line_alpha=1.0,
            sse_guide_alpha=1.0,
            **two,
        )
    for k in range(max(2, min_n)):
        u = k / max(min_n - 1, 1)
        plan.add(
            "duo_sse",
            split_u=1.0,
            pred_p=7.0,
            sse_trail=1.0,
            pred_line_alpha=1.0,
            sse_guide_alpha=1.0,
            sse_min_alpha=smoothstep(u),
            **two,
        )
    plan.hold(
        hold,
        "duo_sse",
        split_u=1.0,
        pred_p=7.0,
        sse_trail=1.0,
        pred_line_alpha=1.0,
        sse_guide_alpha=1.0,
        sse_min_alpha=1.0,
        **two,
    )
    return plan.specs


def plan_ch8_05(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    p = active_profile()
    hold = p.scale_motion(8, 2)
    restore_n = p.scale_motion(24, 6)
    cam_n = p.scale_motion(36, 8)
    trough_n = p.scale_motion(22, 5)
    morph_n = p.scale_motion(36, 8)
    points_n = p.scale_motion(16, 4)
    orbit_n = p.frame_count("orbit_360")
    mark_n = p.scale_motion(14, 4)
    walk_n = p.scale_motion(48, 8)
    plan = _Plan(clip_id)
    keep = {2, 5}
    two = dict(
        dataset="plus",
        n_shown=6,
        alphas=_alphas(_fade_tuple(6, keep, 1.0)),
        plant_pop=1.0,
        split_u=1.0,
        sse_u=1.0,
        sse_trail=1.0,
        pred_p=7.0,
        sse_guide_alpha=1.0,
        sse_min_alpha=1.0,
        pred_line_alpha=1.0,
    )
    plan.hold(hold, "duo_sse", **two)

    # Bring the original plants (and Age = Height/3) back on the 2D side first.
    al_full = (1.0,) * 6
    for k in range(max(2, restore_n)):
        u = k / max(restore_n - 1, 1)
        s = smoothstep(u)
        al = tuple(lerp(a, 1.0, s) for a in _fade_tuple(6, keep, 1.0))
        plan.add(
            "duo_sse",
            dataset="plus",
            n_shown=6,
            alphas=al,
            plant_pop=1.0,
            split_u=1.0,
            sse_u=1.0,
            sse_trail=1.0,
            pred_p=7.0,
            sse_guide_alpha=lerp(1.0, 0.0, s),
            sse_min_alpha=lerp(1.0, 0.0, s),
            pred_line_alpha=lerp(1.0, 0.0, s),
            line_u=s,
            eq_alpha=s,
        )
    restored = dict(
        dataset="plus",
        n_shown=6,
        alphas=al_full,
        plant_pop=1.0,
        split_u=1.0,
        sse_u=1.0,
        sse_trail=1.0,
        line_u=1.0,
        eq_alpha=1.0,
    )
    plan.hold(hold, "duo_sse", **restored)

    # 2D parabola is the intercept–SSE face at slope = 0.
    elev0, az0 = 0.0, -90.0
    elev1, az1 = 28.0, -48.0
    for k in range(max(2, cam_n)):
        u = k / max(cam_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "duo_3d",
            elev=lerp(elev0, elev1, s),
            azim=lerp(az0, az1, s),
            slope_span=s,
            zlim_u=s,
            view_u=s,
            surface_alpha=lerp(0.0, 0.12, s),
            surface_blend=0.0,
            slice_alpha=lerp(1.0, 0.0, s),
            **restored,
        )
    for k in range(max(2, trough_n)):
        u = k / max(trough_n - 1, 1)
        plan.add(
            "duo_3d",
            elev=elev1,
            azim=az1,
            slope_span=1.0,
            zlim_u=1.0,
            view_u=1.0,
            surface_alpha=lerp(0.12, SURFACE_ALPHA_HI, smoothstep(u)),
            surface_blend=0.0,
            slice_alpha=0.0,
            **restored,
        )
    # Morph two-point trough → full-data bowl.
    for k in range(max(2, morph_n)):
        u = k / max(morph_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "duo_3d",
            elev=elev1,
            azim=az1,
            slope_span=1.0,
            zlim_u=1.0,
            view_u=1.0,
            surface_alpha=SURFACE_ALPHA_HI,
            surface_blend=s,
            slice_alpha=0.0,
            **restored,
        )
    three = dict(
        elev=elev1,
        azim=az1,
        slope_span=1.0,
        zlim_u=1.0,
        view_u=1.0,
        surface_alpha=SURFACE_ALPHA_HI,
        surface_blend=1.0,
        slice_alpha=0.0,
        **restored,
    )
    for k in range(max(2, points_n)):
        plan.add("duo_3d", **three)
    spun = {**three, "slice_alpha": 0.0}
    for i in range(max(2, orbit_n)):
        az = az1 + 360.0 * (i / float(orbit_n))
        plan.add("duo_3d", **{**spun, "azim": az})
    for k in range(max(2, mark_n)):
        u = k / max(mark_n - 1, 1)
        plan.add(
            "duo_3d",
            marker_alpha=smoothstep(u),
            marker_u=0.0,
            **spun,
        )
    for k in range(max(2, walk_n)):
        u = k / max(walk_n - 1, 1)
        plan.add(
            "duo_3d",
            marker_alpha=1.0,
            marker_u=smoothstep(u),
            **spun,
        )
    plan.hold(hold, "duo_3d", marker_alpha=1.0, marker_u=1.0, **spun)
    return plan.specs


def plan_ch8_05b(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Shift the intercept (parallel), then tilt the slope around the centroid."""
    p = active_profile()
    hold = p.scale_motion(8, 2)
    sweep_n = p.scale_motion(36, 8)
    back_n = p.scale_motion(18, 5)
    plan = _Plan(clip_id)
    end1 = _end_of_01()
    xbar, ybar = 18.0, 6.0
    plan.hold(hold, "scatter", sse_badge=1.0, **end1)

    def _add_line(b0: float, b1: float, caption: str, cap_a: float = 1.0) -> None:
        plan.add(
            "scatter",
            dataset="core",
            n_shown=5,
            plant_pop=1.0,
            line_u=1.0,
            b0=b0,
            b1=b1,
            eq_alpha=0.0,
            resid_sticks=1.0,
            sse_badge=1.0,
            story_caption=caption,
            story_alpha=cap_a,
        )

    b0_path = [0.0, 1.4, -0.7, 0.0]
    for a, b in zip(b0_path, b0_path[1:]):
        for k in range(max(2, sweep_n if abs(b - a) > 0.5 else back_n)):
            u = k / max(sweep_n - 1, 1) if abs(b - a) > 0.5 else k / max(back_n - 1, 1)
            _add_line(lerp(a, b, smoothstep(u)), 1.0 / 3.0, "intercept moves the line up and down")
    plan.hold(
        hold,
        "scatter",
        sse_badge=1.0,
        resid_sticks=1.0,
        story_caption="intercept moves the line up and down",
        story_alpha=1.0,
        **_end_quiet(),
    )
    m_path = [1.0 / 3.0, 0.08, 0.58, 1.0 / 3.0]
    for a, b in zip(m_path, m_path[1:]):
        n = sweep_n if abs(b - a) > 0.15 else back_n
        for k in range(max(2, n)):
            u = k / max(n - 1, 1)
            m = lerp(a, b, smoothstep(u))
            _add_line(ybar - m * xbar, m, "slope tilts the line around the mean")
    plan.hold(
        hold,
        "scatter",
        sse_badge=1.0,
        story_caption="slope tilts the line around the mean",
        story_alpha=1.0,
        **_end_quiet(),
    )
    return plan.specs


def plan_ch8_05c(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """A far plant pulls the line; a central plant barely does."""
    p = active_profile()
    hold = p.scale_motion(10, 3)
    pop = p.scale_motion(14, 4)
    refit_n = p.scale_motion(24, 6)
    fade_n = p.scale_motion(16, 5)
    plan = _Plan(clip_id)
    end1 = _end_of_01()
    plan.hold(hold, "scatter", sse_badge=1.0, **end1)

    quiet = _end_quiet()

    def _scene(h: float, a: float, extra_u: float, refit: float, caption: str, cap_a: float) -> None:
        plan.add(
            "scatter",
            extra_h=h,
            extra_a=a,
            extra_alpha=extra_u,
            refit=refit,
            resid_sticks=max(extra_u, refit),
            sse_badge=1.0,
            story_caption=caption,
            story_alpha=cap_a,
            **quiet,
        )

    for k in range(max(2, pop)):
        u = k / max(pop - 1, 1)
        _scene(33.0, 2.0, smoothstep(u), 0.0, "a far plant", smoothstep(u))
    for k in range(max(2, refit_n)):
        u = k / max(refit_n - 1, 1)
        _scene(33.0, 2.0, 1.0, smoothstep(u), "far away  —  pulls hard", 1.0)
    plan.hold(
        hold,
        "scatter",
        extra_h=33.0,
        extra_a=2.0,
        extra_alpha=1.0,
        refit=1.0,
        resid_sticks=1.0,
        sse_badge=1.0,
        story_caption="far away  —  pulls hard",
        story_alpha=1.0,
        **quiet,
    )
    for k in range(max(2, fade_n)):
        u = k / max(fade_n - 1, 1)
        s = smoothstep(u)
        _scene(33.0, 2.0, 1.0 - s, 1.0 - s, "", 0.0)
    plan.hold(max(2, hold // 2), "scatter", sse_badge=1.0, **end1)
    for k in range(max(2, pop)):
        u = k / max(pop - 1, 1)
        _scene(18.0, 11.5, smoothstep(u), 0.0, "a middle plant", smoothstep(u))
    for k in range(max(2, refit_n)):
        u = k / max(refit_n - 1, 1)
        _scene(18.0, 11.5, 1.0, smoothstep(u), "near the center  —  barely tilts", 1.0)
    plan.hold(
        hold,
        "scatter",
        extra_h=18.0,
        extra_a=11.5,
        extra_alpha=1.0,
        refit=1.0,
        resid_sticks=1.0,
        sse_badge=1.0,
        story_caption="near the center  —  barely tilts",
        story_alpha=1.0,
        **quiet,
    )
    return plan.specs


def _plan_points_in(clip_id: str, dataset: str, ctx: dict[str, Any] | None = None) -> list[FrameSpec]:
    p = active_profile()
    hold = p.scale_motion(8, 2)
    live = plant_datasets()[dataset]
    n_new_est = max(1, live.n - live.n_base)
    pop = p.scale_motion(10 if n_new_est <= 14 else 5, 3)
    fit_n = p.scale_motion(22, 5)
    plan = _Plan(clip_id)
    pack = (ctx or {}).get("_hita_pack") or {}
    ds = (pack.get("datasets") or {}).get(dataset) or {}
    n_base = int(ds.get("n_base", live.n_base))
    n_total = int(ds.get("n", live.n))
    prev = {"quad": "plus", "nquad": "quad", "hetero": "nquad"}[dataset]
    # Start from previous dataset with its OLS line.
    plan.hold(
        hold,
        "scatter",
        dataset=prev,
        n_shown=n_base,
        line_u=1.0,
        fit_blend=1.0,
        plant_pop=1.0,
        use_ols=1.0,
    )
    for i in range(n_base, n_total):
        for k in range(max(2, pop)):
            u = k / max(pop - 1, 1)
            plan.add(
                "scatter",
                dataset=dataset,
                n_shown=i + 1,
                guide_index=i,
                guide_u=smoothstep(min(1.0, u * 1.2)),
                plant_pop=smoothstep(u),
                line_u=1.0,
                use_ols=1.0,
                fit_blend=0.0,  # still the previous-stage line
            )
    for k in range(max(2, fit_n)):
        u = k / max(fit_n - 1, 1)
        plan.add(
            "scatter",
            dataset=dataset,
            n_shown=n_total,
            plant_pop=1.0,
            line_u=1.0,
            use_ols=1.0,
            fit_blend=smoothstep(u),
        )
    plan.hold(
        hold,
        "scatter",
        dataset=dataset,
        n_shown=n_total,
        plant_pop=1.0,
        line_u=1.0,
        use_ols=1.0,
        fit_blend=1.0,
    )
    return plan.specs


def _resid_callout_index(height, resid) -> int:
    """Largest |residual| in the interior of the plot (not the x-edge)."""
    h = np.asarray(height, dtype=float)
    r = np.asarray(resid, dtype=float)
    if r.size == 0:
        return 0
    interior = (h >= 12.0) & (h <= 30.0)
    cand = np.where(interior)[0] if np.any(interior) else np.arange(r.size)
    return int(cand[np.argmax(np.abs(r[cand]))])


def _plan_residual(
    clip_id: str,
    dataset: str,
    ctx: dict[str, Any] | None = None,
    *,
    label_residual: bool = False,
) -> list[FrameSpec]:
    p = active_profile()
    hold = p.scale_motion(8, 2)
    stick_n = p.scale_motion(16, 4)
    morph_n = p.scale_motion(36, 8)
    plan = _Plan(clip_id)
    live = plant_datasets()[dataset]
    pack = (ctx or {}).get("_hita_pack") or {}
    ds = (pack.get("datasets") or {}).get(dataset) or {}
    n_total = int(ds.get("n", live.n))
    base = dict(
        dataset=dataset,
        n_shown=n_total,
        plant_pop=1.0,
        line_u=1.0,
        use_ols=1.0,
        fit_blend=1.0,
    )
    if label_residual:
        resid = pack.get("resid", {}).get(dataset)
        height = ds.get("h", live.height)
        if resid is None:
            a, b = ols_fit(live.height, live.age)
            resid = residuals(live.height, live.age, a, b)
        base["resid_idx"] = _resid_callout_index(height, resid)
    plan.hold(hold, "residual", residual_u=0.0, residual_sticks=0.0, resid_label=0.0, **base)
    for k in range(max(2, stick_n)):
        u = k / max(stick_n - 1, 1)
        s = smoothstep(u)
        plan.add(
            "residual",
            residual_u=0.0,
            residual_sticks=s,
            resid_label=s if label_residual else 0.0,
            **base,
        )
    for k in range(max(2, morph_n)):
        u = k / max(morph_n - 1, 1)
        plan.add(
            "residual",
            residual_u=smoothstep(u),
            residual_sticks=1.0,
            resid_label=1.0 if label_residual else 0.0,
            **base,
        )
    plan.hold(
        hold,
        "residual",
        residual_u=1.0,
        residual_sticks=1.0,
        resid_label=1.0 if label_residual else 0.0,
        **base,
    )
    return plan.specs


def plan_ch8_06(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    return _plan_points_in(clip_id, "quad", ctx)


def plan_ch8_07(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    return _plan_residual(clip_id, "quad", ctx, label_residual=True)


def plan_ch8_08(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    return _plan_points_in(clip_id, "nquad", ctx)


def plan_ch8_09(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    return _plan_residual(clip_id, "nquad", ctx)


def plan_ch8_10(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    return _plan_points_in(clip_id, "hetero", ctx)


def plan_ch8_11(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    return _plan_residual(clip_id, "hetero", ctx)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def _style_2d(ax, *, xlim, ylim, xlabel, ylabel, font_size, axis_label_size, axes_alpha: float = 1.0) -> None:
    aa = float(np.clip(axes_alpha, 0.0, 1.0))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=axis_label_size, labelpad=16)
    ax.set_ylabel(ylabel, fontsize=axis_label_size, labelpad=16)
    ax.set_title("")
    ax.tick_params(labelsize=font_size, pad=8, width=SPINE_LW, length=10)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    for sp in ax.spines.values():
        sp.set_color("#444444")
        sp.set_linewidth(SPINE_LW)
        sp.set_alpha(aa)
    x0, x1 = float(xlim[0]), float(xlim[1])
    ax.set_xticks([t for t in (0, 6, 12, 18, 24, 30, 36) if x0 - 0.05 <= t <= x1 + 0.05])
    y0, y1 = float(ylim[0]), float(ylim[1])
    if y0 >= -0.05:
        ax.set_yticks([t for t in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18) if y0 - 0.05 <= t <= y1 + 0.05])
    else:
        lo = int(np.floor(y0))
        hi = int(np.ceil(y1))
        step = 2 if (hi - lo) > 8 else 1
        ax.set_yticks(list(range(lo, hi + 1, step)))
    ax.xaxis.label.set_alpha(aa)
    ax.yaxis.label.set_alpha(aa)
    ax.tick_params(colors="#444444")
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_alpha(aa)
    for line in list(ax.xaxis.get_ticklines()) + list(ax.yaxis.get_ticklines()):
        line.set_alpha(aa)
    if aa < 0.02:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)


def _draw_plants(ax, xs, ys, alphas, icon, *, pop: float, zorder: int = 6) -> None:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(alphas) < len(xs):
        alphas = list(alphas) + [1.0] * (len(xs) - len(alphas))
    n = len(xs)
    for i in (range(n - 1) if n else []):
        a = float(alphas[i])
        if a < 0.02:
            continue
        ax.scatter([xs[i]], [ys[i]], s=POINT_S, c=DOT_COLOR, zorder=zorder, alpha=a, linewidths=0)
        _plant_at(ax, xs[i], ys[i], icon, alpha=a, zoom=PLANT_ZOOM, zorder=zorder + 1)
    if n:
        a = float(alphas[n - 1]) * float(pop)
        if a >= 0.02:
            ax.scatter([xs[-1]], [ys[-1]], s=POINT_S, c=DOT_COLOR, zorder=zorder, alpha=min(a, 1.0), linewidths=0)
            _plant_at(ax, xs[-1], ys[-1], icon, alpha=min(a, 1.0), zoom=PLANT_ZOOM * max(pop, 0.15), zorder=zorder + 1)


def _plant_at(ax, x, y, icon, *, alpha: float, zoom: float, zorder: int) -> None:
    im = OffsetImage(icon_with_alpha(icon, alpha), zoom=zoom)
    ab = AnnotationBbox(im, (float(x), float(y)), frameon=False, pad=0.0, zorder=zorder)
    ab.set_clip_on(False)
    ax.add_artist(ab)


def _measure_layout(width_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plant x-centers, visual heights, and icon widths (data units) for clip 00."""
    wr = max(float(width_ratio), 0.05)
    hs = MEASURE_MAX_H * np.asarray(HEIGHTS, dtype=float) / float(HEIGHTS[-1])
    ws = hs * wr
    xs = np.empty(len(hs), dtype=float)
    x = MEASURE_LEFT
    for i, w in enumerate(ws):
        xs[i] = x + 0.5 * float(w)
        x += float(w) + MEASURE_GAP
    return xs, hs, ws


def _data_height_to_zoom(ax, data_h: float, plant_span_px: float) -> float:
    """OffsetImage zoom so the stem-to-bud span equals `data_h` on this axes."""
    bbox = ax.get_position()
    ax_h_in = float(bbox.height) * float(ax.figure.get_size_inches()[1])
    y0, y1 = ax.get_ylim()
    span = float(y1 - y0)
    if span <= 1e-9 or data_h <= 1e-9 or plant_span_px <= 1e-9:
        return 0.04
    plant_in = float(data_h) / span * ax_h_in
    return max(0.04, plant_in * 72.0 / float(plant_span_px))


def _plant_standing(
    ax,
    x,
    ground,
    data_h,
    icon,
    *,
    alpha: float,
    zorder: int,
    plant_span_px: float,
    stem_align: float,
) -> None:
    if alpha < 0.02 or data_h <= 1e-4:
        return
    zoom = _data_height_to_zoom(ax, data_h, plant_span_px)
    im = OffsetImage(icon_with_alpha(icon, alpha), zoom=zoom)
    ab = AnnotationBbox(
        im,
        (float(x), float(ground)),
        frameon=False,
        pad=0.0,
        zorder=zorder,
        box_alignment=(0.5, float(stem_align)),
    )
    ab.set_clip_on(False)
    ax.add_artist(ab)


def _square_bracket(
    ax,
    x_spine: float,
    y0: float,
    y1: float,
    *,
    cap: float,
    u: float,
    alpha: float,
    label: str = "",
    label_alpha: float = 0.0,
) -> None:
    """`[` dimension bracket to the left of the plant; grows with u."""
    su = float(np.clip(u, 0.0, 1.0))
    a = float(np.clip(alpha, 0.0, 1.0))
    if su <= 0.01 or a < 0.02:
        return
    y_top = lerp(float(y0), float(y1), su)
    cap_u = min(1.0, su * 2.4)
    cap_len = float(cap) * cap_u
    kw = dict(color=BRACKET_COLOR, lw=BRACKET_LW, alpha=a, zorder=10, solid_capstyle="butt", clip_on=False)
    ax.plot([x_spine, x_spine], [y0, y_top], **kw)
    ax.plot([x_spine, x_spine + cap_len], [y0, y0], **kw)
    ax.plot([x_spine, x_spine + cap_len], [y_top, y_top], **kw)
    la = a * float(np.clip(label_alpha, 0.0, 1.0))
    if label and la > 0.02:
        ax.text(
            float(x_spine) + 0.5 * cap_len,
            y_top + 0.16,
            label,
            color=BRACKET_COLOR,
            fontsize=_afs(15),
            ha="center",
            va="bottom",
            alpha=la,
            zorder=11,
            clip_on=False,
        )


def _measure_panel(ax, pack, spec) -> None:
    scene_a = float(np.clip(spec.get("scene_alpha", 1.0), 0.0, 1.0))
    ax.set_xlim(*MEASURE_XLIM)
    ax.set_ylim(*MEASURE_YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    for sp in ax.spines.values():
        sp.set_visible(False)
    x0, x1 = MEASURE_XLIM
    if scene_a >= 0.02:
        ax.plot(
            [x0 + 0.25, x1 - 0.25],
            [MEASURE_GROUND, MEASURE_GROUND],
            color="#6d4c41",
            lw=3.0,
            zorder=2,
            solid_capstyle="round",
            clip_on=False,
            alpha=scene_a,
        )

    icon = pack.get("plant_icon_lg", pack["plant_icon"])
    span_px = float(pack.get("plant_span_px") or (icon.shape[0] * ICON_PLANT_FRAC))
    stem_align = float(pack.get("plant_stem_align", ICON_BASE_ALIGN))
    icon_w = float(pack.get("plant_icon_w") or icon.shape[1])
    xs, hs, ws = _measure_layout(icon_w / max(span_px, 1.0))
    n_shown = int(spec.get("n_shown", 0))
    pop = float(np.clip(spec.get("plant_pop", 1.0), 0.0, 1.0))
    n = min(n_shown, len(xs))
    for i in range(n):
        h = float(hs[i])
        a = scene_a
        if i == n - 1:
            a = scene_a * pop
            h = h * max(pop, 0.12)
        _plant_standing(
            ax,
            float(xs[i]),
            MEASURE_GROUND,
            h,
            icon,
            alpha=a,
            zorder=6,
            plant_span_px=span_px,
            stem_align=stem_align,
        )

    if n_shown <= 0 or scene_a < 0.02:
        return
    i = int(np.clip(spec.get("bracket_i", 0), 0, len(xs) - 1))
    u = float(np.clip(spec.get("bracket_u", 0.0), 0.0, 1.0))
    from_i = spec.get("bracket_from")

    def _spine(px: float, pw: float) -> float:
        return px - 0.5 * pw - BRACKET_PAD

    if from_i is None:
        px, ph, pw = float(xs[i]), float(hs[i]), float(ws[i])
        x_spine = _spine(px, pw)
        y1 = MEASURE_GROUND + ph * BRACKET_H_SCALE
        grow_u = u
        label = f"{int(HEIGHTS[i])} cm"
    else:
        j = int(np.clip(from_i, 0, len(xs) - 1))
        su = smoothstep(u)
        px = lerp(float(xs[j]), float(xs[i]), su)
        ph = lerp(float(hs[j]), float(hs[i]), su)
        pw = lerp(float(ws[j]), float(ws[i]), su)
        x_spine = _spine(px, pw)
        y1 = MEASURE_GROUND + ph * BRACKET_H_SCALE
        grow_u = 1.0
        label = f"{int(HEIGHTS[i])} cm"
    _square_bracket(
        ax,
        x_spine,
        MEASURE_GROUND,
        y1,
        cap=BRACKET_CAP,
        u=grow_u,
        alpha=scene_a,
        label=label,
        label_alpha=float(spec.get("label_alpha", 0.0)),
    )


def _guides(ax, x, y, u: float) -> None:
    if u <= 0.01:
        return
    su = float(np.clip(u, 0.0, 1.0))
    ax.plot([x, x], [0.0, y * su], color=GUIDE_COLOR, ls=":", lw=GUIDE_LW, zorder=3)
    ax.plot([0.0, x * su], [y, y], color=GUIDE_COLOR, ls=":", lw=GUIDE_LW, zorder=3)


def _line_through(ax, intercept: float, slope: float, u: float, xlim, *, lw=LINE_LW, color=LINE_COLOR, zorder=4):
    if u <= 0.01:
        return
    x0, x1 = float(xlim[0]), float(xlim[1])
    x_end = lerp(x0, x1, float(np.clip(u, 0.0, 1.0)))
    xs = np.array([x0, x_end])
    ax.plot(xs, intercept + slope * xs, color=color, lw=lw, zorder=zorder, solid_capstyle="round")


def _line_params(pack, spec, xs, ys) -> tuple[float, float]:
    if spec.get("b0") is not None and spec.get("b1") is not None:
        intercept, slope = float(spec["b0"]), float(spec["b1"])
    else:
        intercept, slope = _intercept_slope(
            pack,
            str(spec.get("dataset", "core")),
            float(spec.get("use_ols", 0.0)),
            float(spec.get("fit_blend", 1.0)),
        )
    extra_a = float(spec.get("extra_alpha", 0.0))
    refit = float(np.clip(spec.get("refit", 0.0), 0.0, 1.0))
    if extra_a > 0.02 and refit > 0.01 and spec.get("extra_h") is not None:
        x2 = np.append(np.asarray(xs, dtype=float), float(spec["extra_h"]))
        y2 = np.append(np.asarray(ys, dtype=float), float(spec["extra_a"]))
        a2, b2 = ols_fit(x2, y2)
        intercept = lerp(intercept, a2, refit)
        slope = lerp(slope, b2, refit)
    return intercept, slope


def _inches_per_unit(ax) -> tuple[float, float]:
    bbox = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (bbox.width * fw) / max(x1 - x0, 1e-9), (bbox.height * fh) / max(y1 - y0, 1e-9)


def _linreg_overlays(ax, pack, spec, xs, ys, intercept, slope, xlim, ylim, alphas) -> None:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    _draw_extra_plant(ax, pack, spec)
    _slope_triangle(ax, intercept, slope, spec)
    _intercept_mark(ax, intercept, slope, spec, xlim)
    _centroid_mark(ax, xs, ys, spec)
    _mean_hline(ax, xs, ys, spec, xlim)
    _resid_overlay(ax, xs, ys, intercept, slope, spec, alphas)
    _yhat_marks(ax, xs, ys, intercept, slope, spec, alphas)
    _predict_callout(ax, pack, intercept, slope, spec, xlim, ylim)
    _sse_badge(ax, xs, ys, intercept, slope, spec, xlim, ylim)
    _r2_badge(ax, xs, ys, intercept, slope, spec, xlim, ylim)
    _sad_badge(ax, xs, ys, intercept, slope, spec, xlim, ylim)
    cap = str(spec.get("story_caption") or "")
    cap_a = float(spec.get("story_alpha", 0.0))
    if cap and cap_a > 0.02:
        ax.text(
            0.50,
            0.96,
            cap,
            transform=ax.transAxes,
            color="#333333",
            fontsize=_afs(14),
            alpha=cap_a,
            zorder=12,
            ha="center",
            va="top",
            clip_on=False,
        )


def _draw_extra_plant(ax, pack, spec) -> None:
    a = float(spec.get("extra_alpha", 0.0))
    if a < 0.02 or spec.get("extra_h") is None:
        return
    x, y = float(spec["extra_h"]), float(spec["extra_a"])
    ax.scatter([x], [y], s=POINT_S, c=MEAN_COLOR, zorder=7, alpha=a, linewidths=0)
    _plant_at(ax, x, y, pack["plant_icon"], alpha=a, zoom=PLANT_ZOOM, zorder=8)


def _slope_triangle(ax, intercept, slope, spec) -> None:
    u = float(np.clip(spec.get("slope_tri", 0.0), 0.0, 1.0))
    if u < 0.02:
        return
    x0, x1 = 12.0, 24.0
    y0 = float(intercept) + float(slope) * x0
    y1 = float(intercept) + float(slope) * x1
    run_u = min(1.0, u / 0.45)
    rise_u = min(1.0, max(0.0, u - 0.40) / 0.45)
    x_run = lerp(x0, x1, run_u)
    ax.plot([x0, x_run], [y0, y0], color=MEAN_COLOR, lw=3.0, alpha=0.95, zorder=7, solid_capstyle="round")
    if rise_u > 0.02:
        y_rise = lerp(y0, y1, rise_u)
        ax.plot([x1, x1], [y0, y_rise], color=MEAN_COLOR, lw=3.0, alpha=0.95, zorder=7, solid_capstyle="round")
    la = float(spec.get("slope_lab", 0.0))
    if la > 0.02:
        ax.text(
            0.5 * (x0 + x1),
            y0 - 0.55,
            r"run $= 12$",
            color=MEAN_COLOR,
            fontsize=_afs(13),
            ha="center",
            va="top",
            alpha=la,
            zorder=9,
            clip_on=False,
        )
        ax.text(
            x1 + 0.45,
            0.5 * (y0 + y1),
            r"rise $= 4$",
            color=MEAN_COLOR,
            fontsize=_afs(13),
            ha="left",
            va="center",
            alpha=la,
            zorder=9,
            clip_on=False,
        )
        ax.text(
            x0 + 0.4,
            y1 + 0.35,
            r"slope $= 4/12 = 1/3$",
            color=LINE_COLOR,
            fontsize=_afs(14),
            ha="left",
            va="bottom",
            alpha=la,
            zorder=9,
            clip_on=False,
        )


def _intercept_mark(ax, intercept, slope, spec, xlim) -> None:
    u = float(spec.get("intercept_u", 0.0))
    if u < 0.02:
        return
    x0 = 0.0
    y0 = float(intercept) + float(slope) * x0
    if x0 < float(xlim[0]) - 0.05:
        return
    ax.scatter([x0], [y0], s=POINT_S * 1.15, c=MEAN_COLOR, zorder=9, alpha=u, linewidths=0)
    ax.plot(
        [x0, x0],
        [0.0, y0],
        color=MEAN_COLOR,
        ls=":",
        lw=2.0,
        alpha=0.7 * u,
        zorder=6,
    )
    ax.text(
        x0 + 0.55,
        y0 + 0.55,
        r"intercept $= 0$",
        color=MEAN_COLOR,
        fontsize=_afs(13),
        ha="left",
        va="bottom",
        alpha=u,
        zorder=10,
        clip_on=False,
    )


def _centroid_mark(ax, xs, ys, spec) -> None:
    u = float(spec.get("centroid_u", 0.0))
    if u < 0.02 or xs.size == 0:
        return
    xbar = float(np.mean(xs))
    ybar = float(np.mean(ys))
    ax.axvline(xbar, color=MEAN_COLOR, ls=":", lw=2.0, alpha=0.55 * u, zorder=3)
    ax.axhline(ybar, color=MEAN_COLOR, ls=":", lw=2.0, alpha=0.55 * u, zorder=3)
    ax.scatter(
        [xbar],
        [ybar],
        s=220,
        marker="D",
        c=MEAN_COLOR,
        edgecolors="white",
        linewidths=1.4,
        zorder=9,
        alpha=u,
    )
    ax.text(
        xbar + 0.7,
        ybar + 0.7,
        rf"$(\bar{{x}},\bar{{y}}) = ({_fmt_num(xbar)}, {_fmt_num(ybar)})$",
        color=MEAN_COLOR,
        fontsize=_afs(13),
        ha="left",
        va="bottom",
        alpha=u,
        zorder=10,
        clip_on=False,
    )


def _mean_hline(ax, xs, ys, spec, xlim) -> None:
    u = float(spec.get("mean_line", 0.0))
    if u < 0.02 or ys.size == 0:
        return
    ybar = float(np.mean(ys))
    ax.plot(
        [float(xlim[0]), float(xlim[1])],
        [ybar, ybar],
        color=MEAN_COLOR,
        ls="--",
        lw=LINE_LW * 0.85,
        alpha=u,
        zorder=4,
    )


def _resid_overlay(ax, xs, ys, intercept, slope, spec, alphas) -> None:
    stick_u = float(spec.get("resid_sticks", 0.0))
    sq_u = float(spec.get("resid_squares", 0.0))
    if (stick_u < 0.02 and sq_u < 0.02) or xs.size == 0:
        return
    al = [float(a) for a in (alphas or [1.0] * len(xs))]
    if len(al) < len(xs):
        al += [1.0] * (len(xs) - len(al))
    extra_a = float(spec.get("extra_alpha", 0.0))
    if extra_a > 0.02 and spec.get("extra_h") is not None:
        xs = np.append(xs, float(spec["extra_h"]))
        ys = np.append(ys, float(spec["extra_a"]))
        al.append(extra_a)
    if spec.get("pred_p") is not None:
        yhat = np.full_like(ys, float(spec["pred_p"]), dtype=float)
    else:
        yhat = float(intercept) + float(slope) * xs
    resid = ys - yhat
    xin, yin = _inches_per_unit(ax)
    for i, (x, y, yh, r) in enumerate(zip(xs, ys, yhat, resid)):
        if float(al[i]) < 0.25 or abs(float(r)) < 0.04:
            continue
        if stick_u > 0.02:
            ax.plot(
                [x, x],
                [y, yh],
                color=MEAN_COLOR,
                lw=2.4,
                alpha=0.85 * stick_u,
                zorder=5,
                solid_capstyle="round",
            )
        if sq_u < 0.02:
            continue
        side_y = abs(float(r)) * sq_u
        side_x = side_y * (yin / xin)
        sign = 1.0 if i % 2 == 0 else -1.0
        x_left = x if sign > 0 else x - side_x
        y_bot = min(float(y), float(yh))
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (x_left, y_bot),
            side_x,
            side_y,
            facecolor=SSE_COLOR,
            edgecolor=SSE_COLOR,
            lw=2.0,
            alpha=0.22 * sq_u,
            zorder=4,
            clip_on=False,
        )
        ax.add_patch(rect)
    lab = float(spec.get("resid_lab", 0.0))
    if lab > 0.02 and spec.get("pred_p") is not None:
        p_now = float(spec["pred_p"])
        sse = (p_now - 6.0) ** 2 + (p_now - 8.0) ** 2
        ax.text(
            7.4,
            12.2,
            rf"SSE $= (p-6)^2 + (p-8)^2 = {sse:.1f}$",
            color=SSE_COLOR,
            fontsize=_afs(12),
            ha="left",
            va="center",
            alpha=lab,
            zorder=10,
            clip_on=False,
        )
    elif lab > 0.02:
        ax.text(
            0.98,
            0.90,
            r"$r = y - \hat{y}$",
            transform=ax.transAxes,
            color=MEAN_COLOR,
            fontsize=_afs(14),
            ha="right",
            va="top",
            alpha=lab,
            zorder=10,
            clip_on=False,
        )
    if float(spec.get("l1_lab", 0.0)) > 0.02 and spec.get("pred_p") is not None:
        p_now = float(spec["pred_p"])
        sad = abs(p_now - 6.0) + abs(p_now - 8.0)
        ax.text(
            7.4,
            11.15,
            rf"SAD $= |p-6| + |p-8| = {sad:.1f}$",
            color=LINE_COLOR,
            fontsize=_afs(12),
            ha="left",
            va="center",
            alpha=float(spec.get("l1_lab", 0.0)),
            zorder=10,
            clip_on=False,
        )


def _predict_callout(ax, pack, intercept, slope, spec, xlim, ylim) -> None:
    if spec.get("pred_x") is None:
        return
    u = float(np.clip(spec.get("pred_u", 0.0), 0.0, 1.0))
    if u < 0.02:
        return
    x = float(spec["pred_x"])
    yhat = float(intercept) + float(slope) * x
    v_u = min(1.0, u / 0.45)
    y_top = lerp(0.0, yhat, v_u)
    ax.plot([x, x], [0.0, y_top], color=GUIDE_COLOR, ls=":", lw=GUIDE_LW + 0.4, zorder=5)
    if u > 0.42:
        h_u = min(1.0, (u - 0.42) / 0.35)
        x_left = lerp(x, float(xlim[0]), h_u)
        ax.plot([x, x_left], [yhat, yhat], color=MEAN_COLOR, ls="--", lw=2.4, zorder=6)
    plant_a = min(1.0, max(0.0, (u - 0.55) / 0.35))
    if plant_a > 0.02:
        ax.scatter([x], [yhat], s=POINT_S, c=MEAN_COLOR, zorder=8, alpha=plant_a, linewidths=0)
        _plant_at(ax, x, yhat, pack["plant_icon"], alpha=plant_a, zoom=PLANT_ZOOM, zorder=9)
        y_lab = f"{yhat:.0f}" if abs(yhat - round(yhat)) < 0.04 else f"{yhat:.1f}"
        ax.text(
            x + 0.7,
            yhat + 0.55,
            rf"$\hat{{y}} = {y_lab}$",
            color=MEAN_COLOR,
            fontsize=_afs(14),
            ha="left",
            va="bottom",
            alpha=plant_a,
            zorder=10,
            clip_on=False,
        )
    del ylim


def _yhat_pts(xs, ys, intercept, slope, spec):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    extra_a = float(spec.get("extra_alpha", 0.0))
    if extra_a > 0.02 and spec.get("extra_h") is not None:
        xs = np.append(xs, float(spec["extra_h"]))
        ys = np.append(ys, float(spec["extra_a"]))
    if spec.get("pred_p") is not None:
        yhat = np.full_like(ys, float(spec["pred_p"]), dtype=float)
    else:
        yhat = float(intercept) + float(slope) * xs
    return xs, ys, yhat


def _yhat_marks(ax, xs, ys, intercept, slope, spec, alphas) -> None:
    u = float(spec.get("yhat_marks", 0.0))
    if u < 0.02 or len(xs) == 0:
        return
    al = [float(a) for a in (alphas or [1.0] * len(xs))]
    extra_a = float(spec.get("extra_alpha", 0.0))
    if extra_a > 0.02 and spec.get("extra_h") is not None:
        al = list(al) + [extra_a]
    xs, ys, yhat = _yhat_pts(xs, ys, intercept, slope, spec)
    labeled = False
    for x, y, yh, a in zip(xs, ys, yhat, al):
        if float(a) < 0.25:
            continue
        r = float(y) - float(yh)
        if abs(r) < 0.04:
            continue
        ax.scatter(
            [x],
            [yh],
            s=POINT_S * 0.85,
            facecolors="white",
            edgecolors=LINE_COLOR,
            linewidths=2.2,
            zorder=8,
            alpha=u,
        )
        if labeled:
            continue
        labeled = True
        ax.text(
            float(x) + 0.7,
            float(yh) - 0.15,
            rf"$\hat{{y}} = {_fmt_num(float(yh))}$",
            color=LINE_COLOR,
            fontsize=_afs(14),
            ha="left",
            va="top",
            alpha=u,
            zorder=10,
            clip_on=False,
        )


def _sse_badge(ax, xs, ys, intercept, slope, spec, xlim, ylim) -> None:
    u = float(spec.get("sse_badge", 0.0))
    if u < 0.02 or xs.size == 0:
        return
    extra_a = float(spec.get("extra_alpha", 0.0))
    x_all = xs
    y_all = ys
    if extra_a > 0.02 and spec.get("extra_h") is not None:
        x_all = np.append(xs, float(spec["extra_h"]))
        y_all = np.append(ys, float(spec["extra_a"]))
    sse = float(np.sum((y_all - (float(intercept) + float(slope) * x_all)) ** 2))
    label = r"SSE $= 0$" if sse < 0.04 else rf"SSE $= {sse:.1f}$"
    ax.text(
        float(xlim[1]) - 0.5,
        float(ylim[1]) - 0.7,
        label,
        color=SSE_COLOR,
        fontsize=_afs(14),
        ha="right",
        va="top",
        alpha=u,
        zorder=10,
        clip_on=False,
    )


def _r2_badge(ax, xs, ys, intercept, slope, spec, xlim, ylim) -> None:
    u = float(spec.get("r2_badge", 0.0))
    if u < 0.02 or xs.size == 0:
        return
    extra_a = float(spec.get("extra_alpha", 0.0))
    x_all, y_all = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if extra_a > 0.02 and spec.get("extra_h") is not None:
        x_all = np.append(x_all, float(spec["extra_h"]))
        y_all = np.append(y_all, float(spec["extra_a"]))
    ybar = float(np.mean(y_all))
    sst = float(np.sum((y_all - ybar) ** 2))
    sse = float(np.sum((y_all - (float(intercept) + float(slope) * x_all)) ** 2))
    r2 = 1.0 if sst < 1e-9 else max(0.0, 1.0 - sse / sst)
    ax.text(
        0.98,
        0.86,
        rf"$R^2 = 1 - {sse:.1f}/{sst:.1f} = {r2:.2f}$",
        transform=ax.transAxes,
        color=LINE_COLOR,
        fontsize=_afs(14),
        ha="right",
        va="top",
        alpha=u,
        zorder=10,
        clip_on=False,
    )
    del xlim, ylim


def _sad_badge(ax, xs, ys, intercept, slope, spec, xlim, ylim) -> None:
    u = float(spec.get("sad_badge", 0.0))
    if u < 0.02 or xs.size == 0:
        return
    extra_a = float(spec.get("extra_alpha", 0.0))
    x_all, y_all = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if extra_a > 0.02 and spec.get("extra_h") is not None:
        x_all = np.append(x_all, float(spec["extra_h"]))
        y_all = np.append(y_all, float(spec["extra_a"]))
    if spec.get("pred_p") is not None:
        yhat = np.full_like(y_all, float(spec["pred_p"]), dtype=float)
    else:
        yhat = float(intercept) + float(slope) * x_all
    sad = float(np.sum(np.abs(y_all - yhat)))
    ax.text(
        float(xlim[1]) - 0.5,
        float(ylim[1]) - 2.0,
        rf"SAD $= {sad:.1f}$",
        color=LINE_COLOR,
        fontsize=_afs(14),
        ha="right",
        va="top",
        alpha=u,
        zorder=10,
        clip_on=False,
    )


def _spread_ages(ages, mu: float, scale: float, ylim) -> np.ndarray:
    out = float(mu) + (np.asarray(ages, dtype=float) - float(mu)) * float(scale)
    y0 = float(ylim[0]) + 0.2
    y1 = float(ylim[1]) - 0.2
    return np.clip(out, y0, y1)


def _shape_ages(pack, spec, key: str, n_show: int) -> np.ndarray:
    ka = str(spec.get("shape_a", "normal") or "normal")
    kb = str(spec.get("shape_b", ka) or ka)
    u = float(np.clip(spec.get("shape_u", 0.0), 0.0, 1.0))
    shapes = pack.get("samples_shapes") or {}
    src_a = shapes.get(ka) or pack["samples"]
    ages_a = np.asarray(src_a[key], dtype=float)[:n_show]
    if n_show <= 0:
        return ages_a
    if u < 0.01 or ka == kb:
        return ages_a
    src_b = shapes.get(kb) or pack["samples"]
    ages_b = np.asarray(src_b[key], dtype=float)[:n_show]
    n = min(ages_a.size, ages_b.size)
    return (1.0 - u) * np.sort(ages_a[:n]) + u * np.sort(ages_b[:n])


def _sample_cloud(ax, pack: dict[str, Any], spec: dict[str, Any]) -> None:
    n = int(spec.get("cloud_n", 0) or 0)
    alpha = float(spec.get("cloud_alpha", 1.0))
    if n <= 0 or alpha < 0.02:
        return
    idx = int(spec.get("cloud_h", 2))
    heights = pack["heights"]
    h = float(heights[idx])
    key = str(int(h)) if float(h).is_integer() else str(h)
    ages = np.asarray(pack["samples"][key])[:n]
    if ages.size == 0:
        return
    mu = float(pack["cond_means"][idx])
    scale = float(spec.get("sigma_mult", 1.0))
    ylim = pack.get("ylim", YLIM)
    ages = _spread_ages(ages, mu, scale, ylim)
    ax.scatter(
        np.full(ages.shape, h),
        ages,
        s=CLOUD_S,
        c=PLANT_GREEN,
        edgecolors="#1b5e20",
        linewidths=0.55,
        alpha=min(1.0, alpha),
        zorder=5,
    )


def _hist_and_bell(
    ax,
    pack: dict[str, Any],
    hist_n,
    bell_alpha,
    mean_alpha,
    hist_bar_u: float,
    *,
    cloud_n: int = 0,
    sigma_mult: float = 1.0,
    spec: dict[str, Any] | None = None,
) -> None:
    spec = spec or {}
    heights = pack["heights"]
    means = pack["cond_means"]
    sigma0 = float(pack["cond_sigma"])
    scale = max(float(sigma_mult), 1e-6)
    sigma = sigma0 * scale
    edges = np.asarray(pack["bin_edges"])
    bar_scale = float(pack["bar_scale"])
    n_total = int(pack["n_samples"])
    hist_n = tuple(int(v) for v in hist_n) if hist_n is not None else (0, 0, 0, 0, 0)
    bell_alpha = tuple(float(v) for v in bell_alpha) if bell_alpha is not None else (0.0,) * 5
    mean_alpha = tuple(float(v) for v in mean_alpha) if mean_alpha is not None else (0.0,) * 5
    bu = float(np.clip(hist_bar_u, 0.0, 1.0))
    x_hi = float(pack.get("xlim", XLIM)[1])
    x_lo = float(pack.get("xlim", XLIM)[0])
    ylim = pack.get("ylim", YLIM)
    cloud_on = int(cloud_n) > 0
    ka = str(spec.get("shape_a", "normal") or "normal")
    kb = str(spec.get("shape_b", ka) or ka)
    su = float(np.clip(spec.get("shape_u", 0.0), 0.0, 1.0))
    shaped = ka != "normal" or kb != "normal" or su > 0.01
    if bu < 0.01:
        return
    for i, h in enumerate(heights):
        key = str(int(h)) if float(h).is_integer() else str(h)
        n_avail = int(np.asarray(pack["samples"][key]).size)
        n_show = max(0, min(int(hist_n[i]), n_avail))
        if n_show <= 0 and bell_alpha[i] < 0.02 and mean_alpha[i] < 0.02:
            continue
        mu = float(means[i])
        ages_raw = _shape_ages(pack, spec, key, n_show) if n_show else np.asarray([])
        ages = _spread_ages(ages_raw, mu, scale, ylim) if n_show else ages_raw
        if n_show > 0 and bu > 0:
            counts, _ = np.histogram(ages, bins=edges)
            bin_h = float(edges[1] - edges[0])
            for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
                if c <= 0:
                    continue
                max_w = max(0.15, x_hi - float(h) - 0.8)
                ax.barh(
                    y=0.5 * (lo + hi),
                    width=min(float(c) * bar_scale * bu, max_w),
                    height=bin_h * 0.98,
                    left=float(h),
                    color=HIST_COLOR,
                    edgecolor="#2e7d32",
                    linewidth=0.35,
                    alpha=0.82,
                    zorder=2,
                    align="center",
                )
            if not cloud_on:
                jitter = 0.16 * np.sin(np.asarray(ages, dtype=float) * 17.0)
                ax.scatter(
                    float(h) + jitter,
                    ages,
                    s=14,
                    c=PLANT_GREEN,
                    alpha=0.22 * bu,
                    zorder=3,
                    linewidths=0,
                )
        if bell_alpha[i] > 0.02:
            if shaped:
                y_lo = float(ylim[0]) + 0.05
                y_hi = float(ylim[1]) - 0.05
            else:
                y_lo = max(float(ylim[0]) + 0.05, mu - 3.2 * sigma)
                y_hi = min(float(ylim[1]) - 0.05, mu + 3.2 * sigma)
            ys = np.linspace(y_lo, y_hi, 220)
            pdf = (1.0 - su) * shape_pdf(ys, mu, ka, sigma=sigma) + su * shape_pdf(
                ys, mu, kb, sigma=sigma
            )
            peak = float(np.max(pdf)) if pdf.size else 0.0
            if peak > 1e-12:
                keep = np.where(pdf >= 0.012 * peak)[0]
                if keep.size:
                    i0 = max(0, int(keep[0]) - 2)
                    i1 = min(pdf.size - 1, int(keep[-1]) + 2)
                    ys = ys[i0 : i1 + 1]
                    pdf = pdf[i0 : i1 + 1]
            amp = float(n_total) * float(edges[1] - edges[0]) * bar_scale
            xs = np.minimum(float(h) + amp * pdf, x_hi - 0.5)
            ax.fill_betweenx(ys, float(h), xs, color=BELL_COLOR, alpha=0.22 * bell_alpha[i], zorder=3)
            ax.plot(xs, ys, color=BELL_COLOR, lw=BELL_LW, alpha=bell_alpha[i], zorder=4)
        if mean_alpha[i] > 0.02:
            mu = float(means[i])
            other = float(sum(mean_alpha) - mean_alpha[i])
            x_right = float(h) + 5.2
            if i == 2 and other < 0.08:
                ax.axhline(mu, color=MEAN_COLOR, ls="--", lw=2.2, alpha=mean_alpha[i], zorder=4)
                ax.text(
                    x_lo + 0.7,
                    mu + 0.45,
                    r"average age $= 7$",
                    color=MEAN_COLOR,
                    fontsize=_afs(14),
                    alpha=mean_alpha[i],
                    zorder=7,
                    ha="left",
                    va="bottom",
                    clip_on=False,
                )
            else:
                ax.plot(
                    [float(h) - 0.25, x_right],
                    [mu, mu],
                    color=MEAN_COLOR,
                    ls="--",
                    lw=2.1,
                    alpha=mean_alpha[i],
                    zorder=4,
                )


def _intercept_slope(pack, dataset: str, use_ols: float, fit_blend: float) -> tuple[float, float]:
    """Lerp between the canonical Age = Height/3 line and the dataset OLS.

    ``fit_blend=0`` keeps the *previous* stage OLS (or the perfect 1/3 line for
    core/plus); ``fit_blend=1`` uses this dataset's OLS.
    """
    perfect = (0.0, 1.0 / 3.0)
    fits = pack["fits"]
    prev = {"core": "core", "plus": "core", "quad": "plus", "nquad": "quad", "hetero": "nquad"}[dataset]
    a0, b0 = fits[prev] if dataset not in ("core",) else perfect
    if dataset == "plus" and use_ols < 0.5:
        a0, b0 = perfect
    if dataset == "core":
        a0, b0 = perfect
    a1, b1 = fits[dataset]
    if dataset in ("core", "plus") and use_ols < 0.5:
        a1, b1 = perfect
    u = float(np.clip(fit_blend, 0.0, 1.0))
    if dataset in ("core", "plus"):
        # Clip 1–3 always want Age = Height/3 (use_ols ignored unless we refit).
        if use_ols < 0.5:
            return perfect
        return lerp(a0, a1, u), lerp(b0, b1, u)
    return lerp(a0, a1, u), lerp(b0, b1, u)


def _scatter_panel(ax, pack, spec) -> None:
    ds_name = str(spec.get("dataset", "core"))
    ds = pack["datasets"][ds_name]
    n_shown = int(spec.get("n_shown", 0))
    n_shown = max(0, min(n_shown, int(ds["n"])))
    xs = np.asarray(ds["h"])[:n_shown]
    ys = np.asarray(ds["a"])[:n_shown]
    raw_al = spec.get("alphas")
    if raw_al is None:
        alphas = [1.0] * n_shown
    else:
        alphas = [float(a) for a in raw_al[:n_shown]]
        if len(alphas) < n_shown:
            alphas += [1.0] * (n_shown - len(alphas))
    _draw_plants(ax, xs, ys, alphas, pack["plant_icon"], pop=float(spec.get("plant_pop", 1.0)))
    gi = int(spec.get("guide_index", -1))
    if 0 <= gi < n_shown:
        _guides(ax, float(xs[gi]), float(ys[gi]), float(spec.get("guide_u", 0.0)))
    cloud_n = int(spec.get("cloud_n", 0) or 0)
    _hist_and_bell(
        ax,
        pack,
        spec.get("hist_n"),
        spec.get("bell_alpha"),
        spec.get("mean_alpha"),
        float(spec.get("hist_bar_u", 0.0)),
        cloud_n=cloud_n,
        sigma_mult=float(spec.get("sigma_mult", 1.0)),
        spec=spec,
    )
    cap = str(spec.get("dist_caption") or "")
    cap_a = float(spec.get("dist_caption_alpha", 0.0))
    if cap and cap_a > 0.02:
        xlim, ylim = _xy_lims(pack, spec)
        ax.text(
            float(xlim[1]) - 0.6,
            float(ylim[1]) - 1.25,
            cap,
            color=BELL_COLOR,
            fontsize=_afs(14),
            alpha=cap_a,
            zorder=8,
            ha="right",
            va="bottom",
            clip_on=False,
        )
    _sample_cloud(ax, pack, spec)
    xlim, ylim = _xy_lims(pack, spec)
    intercept, slope = _line_params(pack, spec, xs, ys)
    line_u = float(spec.get("line_u", 0.0))
    if line_u > 0.01:
        _line_through(ax, intercept, slope, line_u, xlim)
        eq_a = float(spec.get("eq_alpha", 0.0))
        if eq_a > 0.02 and abs(slope - 1.0 / 3.0) < 0.02 and abs(intercept) < 0.25:
            ax.text(
                float(xlim[0]) + 1.0,
                float(ylim[1]) - 1.25,
                r"Age $= \frac{1}{3}$ Height",
                color=LINE_COLOR,
                fontsize=_afs(16),
                alpha=eq_a,
                zorder=8,
                ha="left",
                va="bottom",
                clip_on=False,
            )
    pred_a = float(spec.get("pred_line_alpha", 0.0))
    has_pred = spec.get("pred_p") is not None
    if has_pred and pred_a < 0.02:
        pred_a = 1.0
    if pred_a > 0.02:
        p_now = float(spec.get("pred_p", 7.0))
        ax.axhline(p_now, color=MEAN_COLOR, ls="--", lw=2.2, alpha=pred_a, zorder=4)
        # Residual sticks from the two height-18 plants to the prediction.
        for age in (6.0, 8.0):
            ax.plot(
                [X0_SLICE, X0_SLICE],
                [age, p_now],
                color=MEAN_COLOR,
                lw=2.0,
                alpha=0.55 * pred_a,
                zorder=3,
                solid_capstyle="round",
            )
        p_lab = f"{p_now:.0f}" if abs(p_now - round(p_now)) < 0.04 else f"{p_now:.1f}"
        ax.text(
            20.6,
            p_now + 0.45,
            rf"$p = {p_lab}$",
            color=MEAN_COLOR,
            fontsize=_afs(13),
            alpha=pred_a,
            zorder=7,
            clip_on=False,
        )
    _linreg_overlays(ax, pack, spec, xs, ys, intercept, slope, xlim, ylim, alphas)


def _sse_2d(ax, pack, spec, *, font_size, axis_label_size) -> None:
    x0, x1 = SSE2_XLIM
    ax.set_xlim(x0, x1)
    ax.set_ylim(*SSE2_YLIM_2D)
    ax.set_xlabel(r"$p$ = prediction", fontsize=axis_label_size, labelpad=16)
    ax.set_ylabel("Sum of Squared Errors", fontsize=axis_label_size, labelpad=16)
    ax.set_xticks([5, 6, 7, 8, 9])
    ax.tick_params(labelsize=font_size, pad=8, width=SPINE_LW, length=10)
    for sp in ax.spines.values():
        sp.set_color("#444444")
        sp.set_linewidth(SPINE_LW)
    trail = float(spec.get("sse_trail", spec.get("sse_u", 0.0)))
    p = np.linspace(x0, x1, 400)
    sse = sse_two_point(p)
    n = max(2, int(round(float(np.clip(trail, 0.0, 1.0)) * (len(p) - 1))) + 1)
    if trail > 0.01:
        ax.plot(p[:n], sse[:n], color=SSE_COLOR, lw=SSE_LW, zorder=3)
    if trail > 0.85:
        ax.text(
            7.0,
            11.15,
            r"SSE $= (p - 6)^2 + (p - 8)^2$",
            color=SSE_COLOR,
            fontsize=_afs(11),
            zorder=5,
            ha="center",
            va="center",
            clip_on=False,
        )
    pred_p = spec.get("pred_p")
    if pred_p is not None:
        p_now = float(pred_p)
        z_now = float(sse_two_point(p_now))
        ax.scatter([p_now], [z_now], s=140, c=MEAN_COLOR, zorder=8, clip_on=False)
        ax.axvline(p_now, color=MEAN_COLOR, ls=":", lw=1.6, alpha=0.85, zorder=4)
        ga = float(spec.get("sse_guide_alpha", 0.0))
        ma_now = float(spec.get("sse_min_alpha", 0.0))
        if ga > 0.02 and ma_now < 0.35:
            ax.axhline(z_now, color=MEAN_COLOR, ls=":", lw=2.0, alpha=ga, zorder=4)
            ax.text(
                min(p_now + 0.22, x1 - 0.55),
                z_now + 0.32,
                rf"SSE $= {z_now:.0f}$" if abs(z_now - round(z_now)) < 0.05 else rf"SSE $= {z_now:.1f}$",
                color=MEAN_COLOR,
                fontsize=_afs(11),
                alpha=ga,
                va="bottom",
                ha="left",
                zorder=7,
                clip_on=False,
            )
    ma = float(spec.get("sse_min_alpha", 0.0))
    if ma > 0.02:
        ax.text(
            7.0,
            0.95,
            r"minimum at $p=7$",
            color=MEAN_COLOR,
            fontsize=_afs(12),
            alpha=ma,
            zorder=6,
            ha="center",
            va="center",
            clip_on=False,
        )


def _display_zlim(pack, spec) -> tuple[float, float]:
    zu = float(np.clip(spec.get("zlim_u", 1.0), 0.0, 1.0))
    # Small negative pad so the parabola vertex isn't eaten by the 3D floor pane.
    z0 = (SSE2_YLIM_2D[0] - 0.4, SSE2_YLIM_2D[1])
    z1 = pack["sse_zlim"]
    return (lerp(z0[0], z1[0], zu), lerp(z0[1], z1[1], zu))


def _style_3d(ax, *, elev, azim, slope_span, view_u, font_size, axis_label_size, zlim) -> None:
    su = float(np.clip(slope_span, 0.0, 1.0))
    vu = float(np.clip(view_u, 0.0, 1.0))
    # Opening view matches clip 4 (p-range at slope = 0); then frame the OLS bowl.
    b0_lo = lerp(SSE2_XLIM[0], B0_LO, vu)
    b0_hi = lerp(SSE2_XLIM[1], B0_HI, vu)
    m_lo = lerp(-0.02, M_GRID_LO, su)
    m_hi = lerp(0.02, M_GRID_HI, su)
    ax.set_xlim(b0_lo, b0_hi)
    ax.set_ylim(m_lo, m_hi)
    ax.set_zlim(zlim[0], zlim[1])
    tick_fs = lerp(font_size, font_size * 0.78, vu)
    label_fs = lerp(axis_label_size, axis_label_size * 0.88, vu)
    if vu < 0.18:
        ax.set_xlabel("intercept", fontsize=label_fs, labelpad=22)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_zlabel("Sum of Squared Errors", fontsize=label_fs, labelpad=14)
        ax.set_xticks([5, 6, 7, 8, 9])
        ax.set_zticks([0, 2, 4, 6, 8, 10, 12])
        ax.grid(False)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
    else:
        ax.set_xlabel("intercept", fontsize=label_fs, labelpad=14)
        ax.set_ylabel("slope", fontsize=label_fs, labelpad=14)
        ax.set_zlabel("SSE", fontsize=label_fs, labelpad=18)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_zticks([0, 40, 80, 120, 160])
        try:
            ax.set_proj_type("persp")
        except Exception:
            pass
    ax.tick_params(labelsize=tick_fs, pad=6)
    ax.view_init(elev=float(elev), azim=float(azim))
    # Zoom the opening 3D face so the parabola fills the same panel as clip 4.
    if hasattr(ax, "dist"):
        ax.dist = lerp(4.9, 10.8, vu)
    try:
        ax.set_box_aspect(
            (lerp(0.82, 1.00, vu), lerp(0.028, 1.00, vu), lerp(1.10, 0.90, vu)),
            zoom=lerp(1.62, 1.05, vu),
        )
    except TypeError:
        ax.set_box_aspect((lerp(0.82, 1.00, vu), lerp(0.028, 1.00, vu), lerp(1.10, 0.90, vu)))
    except Exception:
        pass
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
        except Exception:
            pass


def _sse_3d(ax, pack, spec, zlim) -> None:
    B0 = pack["sse_B0"]
    M = pack["sse_M"]
    blend = float(np.clip(spec.get("surface_blend", 0.0), 0.0, 1.0))
    Z = (1.0 - blend) * pack["sse_z_two"] + blend * pack["sse_z_full"]
    Zc = np.clip(Z, zlim[0], zlim[1])
    alpha = float(spec.get("surface_alpha", 0.0))
    if alpha > 0.02:
        stride = max(1, B0.shape[0] // 18)
        ax.plot_surface(
            B0,
            M,
            Zc,
            cmap=SSE_CMAP,
            vmin=zlim[0],
            vmax=zlim[1],
            linewidth=0.12,
            antialiased=True,
            shade=False,
            alpha=alpha,
            rstride=stride,
            cstride=stride,
            edgecolor="#5d5d5d",
            zorder=1,
        )
    slice_a = float(spec.get("slice_alpha", 1.0))
    if slice_a > 0.02:
        vu = float(np.clip(spec.get("view_u", 1.0), 0.0, 1.0))
        b0_lo = lerp(SSE2_XLIM[0], B0_LO, vu)
        b0_hi = lerp(SSE2_XLIM[1], B0_HI, vu)
        b0s = np.linspace(b0_lo, b0_hi, len(pack["b0_grid"]))
        m0 = np.zeros_like(b0s)
        ds = pack["datasets"]["plus"]
        z_slice = (1.0 - blend) * sse_two_point(b0s) + blend * sse_ib(
            ds["h"], ds["a"], b0s, 0.0
        )
        z_slice = np.clip(z_slice, zlim[0] + 0.05, zlim[1])
        ax.plot(b0s, m0, z_slice, color=SSE_COLOR, lw=SSE_LW, alpha=slice_a, zorder=8)

    vu = float(np.clip(spec.get("view_u", 1.0), 0.0, 1.0))
    deco_a = slice_a * max(0.0, 1.0 - vu / 0.22)
    if deco_a > 0.02:
        ma2 = float(spec.get("sse_min_alpha", 0.0)) * deco_a
        if ma2 > 0.02:
            ax.plot([7.0, 7.0], [0.0, 0.0], [zlim[0], min(zlim[1], 12.0)], color=MEAN_COLOR, ls=":", lw=2.2, alpha=ma2)
            ax.scatter([7.0], [0.0], [2.0], s=140, c=MEAN_COLOR, depthshade=False, alpha=ma2, zorder=9)
        ta = float(spec.get("tangent_alpha", 0.0)) * deco_a
        if ta > 0.02:
            ax.plot([5.2, 8.8], [0.0, 0.0], [2.0, 2.0], color=TANGENT_COLOR, lw=2.8, alpha=ta, zorder=8)

    ma = float(spec.get("marker_alpha", 0.0))
    if ma > 0.02:
        u = float(np.clip(spec.get("marker_u", 0.0), 0.0, 1.0))
        b0_s, m_s = MARKER_START
        b0_e, m_e = float(pack["b0_ols"]), float(pack["m_ols"])
        b0_now = lerp(b0_s, b0_e, u)
        m_now = lerp(m_s, m_e, u)
        ds = pack["datasets"]["plus"]
        z_now = float(
            (1.0 - blend) * sse_ib(np.asarray([18.0, 18.0]), np.asarray([6.0, 8.0]), b0_now, m_now)
            + blend * sse_ib(ds["h"], ds["a"], np.asarray(b0_now), np.asarray(m_now))
        )
        z_now = float(np.clip(z_now, zlim[0], zlim[1]))
        ax.plot([b0_now, b0_now], [m_now, m_now], [0.0, z_now], color="#c0392b", lw=2.8, alpha=ma, zorder=9)
        z_vis = float(np.clip(z_now + 0.04 * (zlim[1] - zlim[0]), zlim[0], zlim[1]))
        ax.scatter([b0_now], [m_now], [z_vis], s=240, c="#c0392b", depthshade=False, alpha=ma, zorder=12)
        if u > 0.97:
            ax.text2D(
                0.50,
                0.90,
                "minimum",
                transform=ax.transAxes,
                color="#c0392b",
                fontsize=_afs(12),
                ha="center",
                va="top",
            )


def _left_line_from_marker(ax, pack, spec) -> None:
    ma = float(spec.get("marker_alpha", 0.0))
    if ma < 0.02:
        return
    u = float(np.clip(spec.get("marker_u", 0.0), 0.0, 1.0))
    b0_s, m_s = MARKER_START
    b0_now = lerp(b0_s, float(pack["b0_ols"]), u)
    m_now = lerp(m_s, float(pack["m_ols"]), u)
    _line_through(ax, b0_now, m_now, 1.0, pack["xlim"], color="#c0392b", lw=LINE_LW)


def _residual_callout(ax, spec, xs, y_pts, resid, line_y_at) -> None:
    la = float(spec.get("resid_label", 0.0))
    if la < 0.02:
        return
    idx = spec.get("resid_idx")
    if idx is None:
        return
    i = int(idx)
    if i < 0 or i >= len(xs):
        return
    x = float(xs[i])
    y_obs = float(y_pts[i])
    y_hat = float(line_y_at(x))
    y_mid = y_hat + 0.62 * float(resid[i])
    ax.plot([x, x], [y_obs, y_hat], color=MEAN_COLOR, lw=2.8, alpha=la, zorder=5, solid_capstyle="round")
    ax.annotate(
        r"residual" "\n" r"$r = y - \hat{y}$",
        xy=(x, y_mid),
        xytext=(0.48, 0.90),
        textcoords="axes fraction",
        fontsize=_afs(13),
        color=MEAN_COLOR,
        ha="center",
        va="top",
        linespacing=1.12,
        alpha=la,
        zorder=12,
        clip_on=False,
        arrowprops=dict(
            arrowstyle="-|>",
            color=MEAN_COLOR,
            lw=2.0,
            mutation_scale=14,
            connectionstyle="arc3,rad=0.08",
            shrinkA=6,
            shrinkB=8,
        ),
    )


def _residual_panel(ax, pack, spec, *, font_size, axis_label_size) -> None:
    ds_name = str(spec.get("dataset", "quad"))
    ds = pack["datasets"][ds_name]
    n_shown = int(spec.get("n_shown", ds["n"]))
    xs = np.asarray(ds["h"])[:n_shown]
    ys = np.asarray(ds["a"])[:n_shown]
    a, b = pack["fits"][ds_name]
    r = np.asarray(pack["resid"][ds_name])[:n_shown]
    u = float(np.clip(spec.get("residual_u", 0.0), 0.0, 1.0))
    sticks = float(np.clip(spec.get("residual_sticks", 0.0), 0.0, 1.0))
    fitted = a + b * xs
    # Line drops to y=0; points keep residual distance to the line.
    line_y_at = lambda x: (1.0 - u) * (a + b * x)
    y_pts = line_y_at(xs) + r
    xlim, ylim0 = _xy_lims(pack, spec)
    ylim1 = pack["resid_ylim"][ds_name]
    ylim = (lerp(ylim0[0], ylim1[0], u), lerp(ylim0[1], ylim1[1], u))
    xlabel = "Height (cm)"
    ylabel = "Age (years)" if u < 0.5 else "Residual (years)"
    _style_2d(
        ax,
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
        font_size=font_size,
        axis_label_size=axis_label_size,
    )
    if u > 0.15:
        ax.axhline(0.0, color="#444444", lw=SPINE_LW, zorder=2)
    if sticks > 0.02:
        for x, y0, y1 in zip(xs, y_pts, line_y_at(xs)):
            ax.plot([x, x], [y0, y1], color="#e07a3d", lw=2.0, alpha=0.85 * sticks, zorder=3)
    _line_through(ax, 0.0 if u > 0.98 else a * (1.0 - u), b * (1.0 - u), 1.0, xlim)
    plant_a = max(0.0, 1.0 - u)
    if plant_a > 0.04:
        _draw_plants(ax, xs, y_pts, [plant_a] * len(xs), pack["plant_icon"], pop=1.0)
    colors = np.where(r >= 0, PLANT_GREEN, "#d62728")
    ax.scatter(xs, y_pts, s=POINT_S, c=colors, edgecolors="#222222", linewidths=0.7, zorder=6, alpha=max(u, 0.25))
    _residual_callout(ax, spec, xs, y_pts, r, line_y_at)


def _layout_rects(split_u: float, *, three_d: bool = False) -> tuple[list[float], list[float]]:
    del three_d  # 2D SSE panel and 3D view share one rect so the clip-5 handoff doesn't jump.
    s = smoothstep(float(np.clip(split_u, 0.0, 1.0)))
    left = [
        lerp(SCATTER_RECT[0], DUO_LEFT_X, s),
        AX_BOTTOM,
        lerp(SCATTER_RECT[2], DUO_W, s),
        AX_HEIGHT,
    ]
    right = [
        lerp(1.06, DUO_RIGHT_X, s),
        AX_BOTTOM,
        lerp(DUO_W, DUO_W, s),
        AX_HEIGHT,
    ]
    return left, right


def render_frame(pack: dict[str, Any], spec: dict[str, Any]):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from hita.export.raster import fig_to_image

    kind = str(spec.get("kind", "scatter"))
    dpi = int(pack.get("dpi", 200))
    fs = float(pack.get("font_size", 13.75))
    als = float(pack.get("axis_label_size", 15.0))
    figsize = pack.get("figsize", FIGSIZE)
    fig = plt.figure(figsize=figsize, facecolor="white")

    if kind == "measure":
        ax = fig.add_axes(MEASURE_RECT)
        _measure_panel(ax, pack, spec)
    elif kind == "scatter":
        ax = fig.add_axes(SCATTER_RECT)
        xlim, ylim = _xy_lims(pack, spec)
        _style_2d(
            ax,
            xlim=xlim,
            ylim=ylim,
            xlabel="Height (cm)",
            ylabel="Age (years)",
            font_size=fs,
            axis_label_size=als,
            axes_alpha=float(spec.get("axes_alpha", 1.0)),
        )
        _scatter_panel(ax, pack, spec)
    elif kind == "residual":
        ax = fig.add_axes(SCATTER_RECT)
        _residual_panel(ax, pack, spec, font_size=fs, axis_label_size=als)
    elif kind == "duo_sse":
        left, right = _layout_rects(float(spec.get("split_u", 1.0)))
        ax = fig.add_axes(left)
        ax_r = fig.add_axes(right)
        xlim, ylim = _xy_lims(pack, spec)
        _style_2d(
            ax,
            xlim=xlim,
            ylim=ylim,
            xlabel="Height (cm)",
            ylabel="Age (years)",
            font_size=fs,
            axis_label_size=als,
        )
        _scatter_panel(ax, pack, spec)
        _sse_2d(ax_r, pack, spec, font_size=fs, axis_label_size=als)
    elif kind == "duo_3d":
        left, right = _layout_rects(1.0, three_d=True)
        ax = fig.add_axes(left)
        ax3 = fig.add_axes(right, projection="3d")
        xlim, ylim = _xy_lims(pack, spec)
        _style_2d(
            ax,
            xlim=xlim,
            ylim=ylim,
            xlabel="Height (cm)",
            ylabel="Age (years)",
            font_size=fs,
            axis_label_size=als,
        )
        _scatter_panel(ax, pack, spec)
        _left_line_from_marker(ax, pack, spec)
        zlim = _display_zlim(pack, spec)
        _style_3d(
            ax3,
            elev=float(spec.get("elev", 22.0)),
            azim=float(spec.get("azim", -52.0)),
            slope_span=float(spec.get("slope_span", 1.0)),
            view_u=float(spec.get("view_u", 1.0)),
            font_size=fs,
            axis_label_size=als,
            zlim=zlim,
        )
        _sse_3d(ax3, pack, spec, zlim)
    else:
        raise ValueError(f"unknown ch8 frame kind: {kind!r}")

    return fig_to_image(fig, dpi=dpi)
