"""Chapter 4 — thin wrapper around ``tutorial_template`` defaults."""
from __future__ import annotations

from pathlib import Path

import handwrite_tutorial as hw
from tutorial_template import (
    DEFAULT_EXPORT,
    DEFAULT_LAYOUT,
    DEFAULT_TYPOGRAPHY,
    TUTORIAL_THEMES,
    TutorialComposer,
    TutorialScene,
    list_themes,
    make_composer,
)

OUTPUT_DIR = Path("renders")
OUTPUT_DIR.mkdir(exist_ok=True)

CH4_FIGSIZE = DEFAULT_EXPORT.figsize
CH4_EXPORT_DPI = DEFAULT_EXPORT.dpi
CH4_SAVE_PAD_INCHES = DEFAULT_EXPORT.pad_inches

CH4_HERE_SECTION_TITLE = "We are here"
CH4_NOTATION_SECTION_TITLE = "Notation"
CH4_FORMULAS_SECTION_TITLE = "Formulas"

CH4_RIGHT_SECTION_TITLE = CH4_HERE_SECTION_TITLE
CH4_BOTTOM_SECTION_TITLE = CH4_FORMULAS_SECTION_TITLE

CH4_COMPOSER = make_composer("classic_light")
CH4_TEMPLATE_THEME = "dark_rails"

# ch4_02 end frame fills the canvas; morph shrinks it into the template plot slot.
CH4_LIK_PLOT_START_RECT = (0.0, 0.0, 1.0, 1.0)

# Duo layout: 2D panel −10%, 3D +10%, 3D vertically aligned with 2D (not 2D+knobs).
CH4_DUO_DATA_SCALE = 0.90
CH4_DUO_AX3D_SCALE = 1.10
CH4_DUO_PLOTS_X_SHIFT_MM = -4.0   # nudge 2D + 3D left; knobs unchanged
CH4_DUO_PLOTS_X_SHIFT_PT = CH4_DUO_PLOTS_X_SHIFT_MM * 72.0 / 25.4


def _ch4_fig_x_shift_frac(fig, shift_pt: float) -> float:
    w_pt = float(fig.get_figwidth()) * 72.0
    return float(shift_pt) / w_pt if w_pt > 0 else 0.0


def ch4_duo_plot_layout_tune(
    fig,
    ax_data,
    ax3d,
    *,
    data_scale: float | None = None,
    ax3d_scale: float | None = None,
    x_shift_pt: float | None = None,
) -> None:
    """Shrink 2D, enlarge 3D, align 3D center with 2D; leave knob axes untouched."""
    data_scale = CH4_DUO_DATA_SCALE if data_scale is None else float(data_scale)
    ax3d_scale = CH4_DUO_AX3D_SCALE if ax3d_scale is None else float(ax3d_scale)
    x_shift_pt = CH4_DUO_PLOTS_X_SHIFT_PT if x_shift_pt is None else float(x_shift_pt)
    x_shift = _ch4_fig_x_shift_frac(fig, x_shift_pt)
    fig.canvas.draw()
    d = ax_data.get_position()
    r = ax3d.get_position()
    dw = d.width * data_scale
    dh = d.height * data_scale
    dx = d.x0 + 0.5 * (d.width - dw) + x_shift
    dy = d.y0 + 0.5 * (d.height - dh)
    ax_data.set_position([dx, dy, dw, dh])
    rw = r.width * ax3d_scale
    rh = r.height * ax3d_scale
    cy = dy + 0.5 * dh
    rx = r.x0 + 0.5 * (r.width - rw) + x_shift
    ry = cy - 0.5 * rh
    ax3d.set_position([rx, ry, rw, rh])


def ch4_duo_partial_layout_tune(
    fig,
    ax_data,
    axes_partial,
    *,
    data_scale: float | None = None,
    right_scale: float | None = None,
    x_shift_pt: float | None = None,
    hspace_frac: float = 0.10,
    y_drop_mm: float = 0.0,
    y_lift_mm: float = 0.0,
    x_shift_mm: float = 0.0,
    subplot_height_frac: float = 1.0,
) -> None:
    """Like ``ch4_duo_plot_layout_tune`` but right column is stacked 1-D NLL panels."""
    data_scale = CH4_DUO_DATA_SCALE if data_scale is None else float(data_scale)
    right_scale = CH4_DUO_AX3D_SCALE if right_scale is None else float(right_scale)
    x_shift_pt = CH4_DUO_PLOTS_X_SHIFT_PT if x_shift_pt is None else float(x_shift_pt)
    x_shift = _ch4_fig_x_shift_frac(fig, x_shift_pt)
    fig.canvas.draw()
    d = ax_data.get_position()
    dw = d.width * data_scale
    dh = d.height * data_scale
    dx = d.x0 + 0.5 * (d.width - dw) + x_shift
    dy = d.y0 + 0.5 * (d.height - dh)
    ax_data.set_position([dx, dy, dw, dh])
    rs = [ax.get_position() for ax in axes_partial]
    x0 = min(r.x0 for r in rs)
    y0 = min(r.y0 for r in rs)
    x1 = max(r.x0 + r.width for r in rs)
    y1 = max(r.y0 + r.height for r in rs)
    rw_full = (x1 - x0) * right_scale
    rh_full = (y1 - y0) * right_scale
    cy = dy + 0.5 * dh
    rx = x0 + 0.5 * ((x1 - x0) - rw_full) + x_shift
    ry = cy - 0.5 * rh_full
    fig_h_pt = max(float(fig.get_figheight()) * 72.0, 1e-9)
    fig_w_pt = max(float(fig.get_figwidth()) * 72.0, 1e-9)
    if float(y_drop_mm) != 0.0:
        ry -= float(y_drop_mm) * 72.0 / 25.4 / fig_h_pt
    n = len(axes_partial)
    gap = float(hspace_frac) * rh_full / max(n, 1)
    sub_h = (rh_full - gap * max(n - 1, 0)) / max(n, 1)
    panel_h = sub_h * float(subplot_height_frac)
    top_cap = 0.915

    def _place_partial_stack(anchor_top: float) -> None:
        for i, ax in enumerate(axes_partial):
            yi = anchor_top - (i + 1) * panel_h - i * gap
            ax.set_position([rx, yi, rw_full, panel_h])

    anchor_top = ry + rh_full
    overflow = anchor_top - top_cap
    if overflow > 0.0:
        anchor_top -= overflow
    _place_partial_stack(anchor_top)
    dx_frac = float(x_shift_mm) * 72.0 / 25.4 / fig_w_pt
    dy_frac = float(y_lift_mm) * 72.0 / 25.4 / fig_h_pt
    if dx_frac != 0.0 or dy_frac != 0.0:
        for ax in axes_partial:
            p = ax.get_position()
            ax.set_position([p.x0 + dx_frac, p.y0 + dy_frac, p.width, p.height])


CH4_RAILS_CACHE_3D = "ch4_lik_3d_v13"
CH4_RAILS_CACHE_GD = "ch4_lik_gd_v9"
CH4_RAILS_CACHE_SHELL = "ch4_lik_shell_v1"


def ch4_rails_cache_key_gd(
    *,
    bold_update_idx: int | None = None,
    bold_all: bool = False,
    grad_red: bool = False,
) -> str:
    """Cache key for a GD bottom-rail variant (corner + formulas, static per variant)."""
    if bold_all:
        bold_tag = "all"
    elif bold_update_idx is None:
        bold_tag = "none"
    else:
        bold_tag = str(int(bold_update_idx))
    color_tag = "red" if grad_red else "rgb"
    return f"{CH4_RAILS_CACHE_GD}_{bold_tag}_{color_tag}"

_CH4_CACHED_GD_BOTTOM: list[dict] | None = None
_CH4_CACHED_3D_BOTTOM: list[dict] | None = None
_CH4_CACHED_NOTATION_CORNER: list[dict] | None = None
_CH4_KNOB_ASSET_PACK: tuple | None = None

CH4_LABELED_KNOB_NAMES = ("wst", "wel", "b")
CH4_KNOB_CROP_PAD = (11, 11, 2, 1)   # left, right, bottom, top — matches knob_1_cropped
CH4_KNOB_CROP_ALPHA = 128
CH4_KNOB_CROP_WHITE = 250


def ch4_crop_knob_image(src, *, pad=CH4_KNOB_CROP_PAD, alpha_thr=CH4_KNOB_CROP_ALPHA, white_thr=CH4_KNOB_CROP_WHITE):
    """Tight crop around the dial (non-white opaque pixels), same rules as knob_1_cropped."""
    import numpy as np
    from PIL import Image

    pl, pr, pb, pt = pad
    im = Image.open(src).convert("RGBA") if not isinstance(src, Image.Image) else src.convert("RGBA")
    arr = np.asarray(im)
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = (alpha > alpha_thr) & (rgb.max(axis=2) < white_thr)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return im
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    x0 = max(0, x0 - int(pl))
    x1 = min(im.width - 1, x1 + int(pr))
    y0 = max(0, y0 - int(pt))
    y1 = min(im.height - 1, y1 + int(pb))
    return im.crop((x0, y0, x1 + 1, y1 + 1))


def ch4_ensure_labeled_knob_pngs(*, force: bool = False) -> None:
    for name in CH4_LABELED_KNOB_NAMES:
        src = OUTPUT_DIR / f"knob_{name}.png"
        dst = OUTPUT_DIR / f"knob_{name}_cropped.png"
        if not src.is_file():
            raise FileNotFoundError(f"missing knob source: {src}")
        if dst.is_file() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        ch4_crop_knob_image(src).save(dst)


def ch4_blend_knob_images(old_im, new_im, u: float):
    import numpy as np
    from PIL import Image

    u = float(np.clip(float(u), 0.0, 1.0))
    if u <= 1e-9:
        return old_im.convert("RGBA")
    if u >= 1.0 - 1e-9:
        return new_im.convert("RGBA")
    o = np.asarray(old_im.convert("RGBA"), dtype=np.float32)
    n = np.asarray(new_im.convert("RGBA"), dtype=np.float32)
    if o.shape != n.shape:
        n_im = new_im.convert("RGBA")
        n_im = n_im.resize(old_im.size, Image.Resampling.LANCZOS)
        n = np.asarray(n_im, dtype=np.float32)
    out = (1.0 - u) * o + u * n
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGBA")


def ch4_knob_row_from_blend(numbered_rgbs, labeled_rgbs, blends) -> tuple:
    """Build a 3-tuple of knob RGBA images crossfading numbered → labeled per slot."""
    import numpy as np

    blends = tuple(float(np.clip(float(u), 0.0, 1.0)) for u in blends)
    return tuple(
        ch4_blend_knob_images(numbered_rgbs[i], labeled_rgbs[i], blends[i])
        for i in range(3)
    )


def _ch4_resolve_knob_probe():
    import inspect

    for fr in inspect.stack()[1:]:
        probe = fr.frame.f_globals.get("_ch3_probe_knob_canvas_side")
        if probe is not None:
            return probe
    raise RuntimeError(
        "ch4_knob_asset_pack needs _ch3_probe_knob_canvas_side (run ch3 setup cell first)"
    )


def ch4_knob_asset_pack():
    """Labeled knobs for ch4_04+ (w_ST / w_EL / b dials)."""
    from PIL import Image

    global _CH4_KNOB_ASSET_PACK
    if _CH4_KNOB_ASSET_PACK is not None:
        return _CH4_KNOB_ASSET_PACK
    ch4_ensure_labeled_knob_pngs()
    probe = _ch4_resolve_knob_probe()
    rgbs = tuple(
        Image.open(OUTPUT_DIR / f"knob_{name}_cropped.png").convert("RGBA")
        for name in CH4_LABELED_KNOB_NAMES
    )
    sides = tuple(probe(im) for im in rgbs)
    side_uni = int(max(sides))
    _CH4_KNOB_ASSET_PACK = (rgbs, (side_uni, side_uni, side_uni))
    return _CH4_KNOB_ASSET_PACK


def ch4_knob_asset_pack_blended(numbered_pack, labeled_pack, blends) -> tuple:
    """Canvas sides from numbered pack; pixels blend numbered → labeled per slot."""
    num_rgbs, num_sides = numbered_pack
    lab_rgbs, _ = labeled_pack
    rgbs = ch4_knob_row_from_blend(num_rgbs, lab_rgbs, blends)
    return rgbs, num_sides


# --- Notation / formula blocks for likelihood story clips ---

def _ch4_notation_weights_row(**extra) -> dict:
    """Knob labels — same typography/spacing as the 04+ We are here weights block."""
    row = {
        "block_fs": CH4_HERE_BLOCK_FS,
        "line_dy_pt": CH4_HERE_WEIGHTS_LINE_DY_PT,
        "top_pad_pt": 0.0,
        "label_gap_pt": 0.0,
        "align": "left",
        "pt_units": True,
        "text": "w_ST = knob 1\nw_EL = knob 2\nb = knob 3",
        "role": "weights",
    }
    row.update(extra)
    return row


def _ch4_notation_yi_xi_lines() -> tuple[str, str]:
    return (
        r"$y_{i} = \begin{cases} 1 & \text{if i passed} \\ 0 & \text{if i failed} \end{cases}$",
        r"$x_{i} = \left[x_{i,\mathrm{ST}},\, x_{i,\mathrm{EL}}\right]$",
    )


def _ch4_notation_yi_xi_block(**extra) -> dict:
    yi, xi = _ch4_notation_yi_xi_lines()
    base = dict(
        text=yi + "\n" + xi,
        block_fs=CH4_HERE_BLOCK_FS,
        line_dy_pt=CH4_HERE_LOOSE_LINE_DY_PT,
        align="left",
        top_pad_pt=0.0,
        pt_units=True,
        role="formula",
        text_x_frac=0.04,
        cases_row_gap_pt=CH4_CASES_ROW_GAP_PT,
    )
    base.update(extra)
    return base


def ch4_notation_blocks_basic() -> list[dict]:
    return [_ch4_notation_weights_row()]


def ch4_notation_blocks_expanded() -> list[dict]:
    """Right-rail notation for ch4_03 — same typography as 04+ We are here."""
    yi, xi = _ch4_notation_yi_xi_lines()
    return [{
        "block_fs": CH4_HERE_BLOCK_FS,
        "line_dy_pt": CH4_HERE_WEIGHTS_LINE_DY_PT,
        "top_pad_pt": CH4_NOTATION_EXPANDED_TOP_PAD_PT,
        "label_gap_pt": 0.0,
        "align": "left",
        "pt_units": True,
        "text": (
            "w_ST = knob 1\nw_EL = knob 2\nb = knob 3\n"
            + yi + "\n" + xi
        ),
        "line_y_inset_pt": {CH4_NOTATION_YI_LINE_IDX: CH4_NOTATION_YI_DROP_PT},
        "line_extra_dy_pt": {CH4_NOTATION_YI_LINE_IDX: CH4_NOTATION_YI_XI_GAP_PT},
        "line_body_fs": {CH4_NOTATION_XI_LINE_IDX: CH4_NOTATION_XI_BODY_FS},
        "line_subscript_scale": {CH4_NOTATION_XI_LINE_IDX: CH4_NOTATION_XI_SUB_SCALE},
        "cases_row_gap_pt": CH4_CASES_ROW_GAP_PT,
        "role": "weights",
    }]


def ch4_formula_blocks_likelihood() -> list[dict]:
    return [ch4_formula_col_likelihood()]


def ch4_formula_blocks_log_likelihood() -> list[dict]:
    return [ch4_formula_col_log()]


def ch4_formula_blocks_nll() -> list[dict]:
    return [ch4_formula_nll_block()]


# Navy → blue → purple → magenta → red (matches ch3_loglik_ridge_dark_heatmap.png)
CH4_NLL_HEATMAP_COLORS = (
    "#08111f",
    "#122A88",
    "#2D5BFF",
    "#7A4DFF",
    "#C13CFF",
    "#FF3B5C",
)
CH4_NLL_HEATMAP_PCT_LO = 2.0
CH4_NLL_HEATMAP_PCT_HI = 98.0

_CH4_NLL_HEATMAP_CMAP = None


def ch4_nll_heatmap_cmap():
    """Smooth navy → blue → purple → magenta → red (ch3 ridge dark heatmap)."""
    global _CH4_NLL_HEATMAP_CMAP
    if _CH4_NLL_HEATMAP_CMAP is None:
        from matplotlib.colors import LinearSegmentedColormap

        _CH4_NLL_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
            "ch4_nll_ridge_neon",
            CH4_NLL_HEATMAP_COLORS,
            N=256,
        )
    return _CH4_NLL_HEATMAP_CMAP


def ch4_nll_heatmap_limits(values, *, vmin=None, vmax=None, pct=None):
    """Default vmin/vmax via percentiles (same spirit as ridge export script)."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)
    lo_p, hi_p = (
        (CH4_NLL_HEATMAP_PCT_LO, CH4_NLL_HEATMAP_PCT_HI)
        if pct is None
        else (float(pct[0]), float(pct[1]))
    )
    if vmin is None:
        vmin = float(np.nanpercentile(arr, lo_p))
    if vmax is None:
        vmax = float(np.nanpercentile(arr, hi_p))
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return float(vmin), float(vmax)


def ch4_nll_heatmap_color(nll, vmin, vmax) -> str:
    import matplotlib as mpl
    from matplotlib.colors import Normalize

    lo, hi = float(vmin), float(vmax)
    norm = Normalize(vmin=lo, vmax=hi)
    rgba = ch4_nll_heatmap_cmap()(norm(float(nll)))
    return mpl.colors.to_hex(rgba, keep_alpha=False)


def ch4_nll_heatmap_facecolors(values, *, vmin=None, vmax=None, alpha=1.0, pct=None):
    """RGBA facecolor array for ``plot_surface`` (shape matches ``values`` + alpha channel)."""
    import numpy as np
    from matplotlib.colors import Normalize

    Z = np.asarray(values, dtype=float)
    lo, hi = ch4_nll_heatmap_limits(Z, vmin=vmin, vmax=vmax, pct=pct)
    norm = Normalize(vmin=lo, vmax=hi)
    rgba = ch4_nll_heatmap_cmap()(norm(Z))
    rgba = np.asarray(rgba, dtype=float)
    rgba[..., 3] = float(alpha)
    return rgba


CH4_NLL_HEATMAP_SECTION_TITLE = "NLL color scale"
CH4_NLL_HEATMAP_LEGEND_BAR_WEIGHT = 0.52
CH4_NLL_COLORBAR_W_FRAC = 0.08


def _ch4_nll_heatmap_legend_fmt(nll) -> str:
    return f"{float(nll):.1f}"


def ch4_nll_heatmap_legend_blocks(nll_lo, nll_hi) -> list[dict]:
    """Right-rail legend for NLL-colored voxels / heatmaps (low = good, high = bad)."""
    lo = float(nll_lo)
    hi = float(nll_hi)
    return [{
        "block_fs": CH4_HERE_BLOCK_FS,
        "line_dy_pt": CH4_HERE_WEIGHTS_LINE_DY_PT,
        "top_pad_pt": 0.0,
        "label_gap_pt": 0.0,
        "align": "left",
        "pt_units": True,
        "role": "legend",
        "weight": CH4_NLL_HEATMAP_LEGEND_BAR_WEIGHT,
        "nll_colorbar": True,
        "nll_lo": lo,
        "nll_hi": hi,
        "label_lo": f"{_ch4_nll_heatmap_legend_fmt(lo)}  (better)",
        "label_hi": f"{_ch4_nll_heatmap_legend_fmt(hi)}  (worse)",
    }]


def ch4_draw_nll_colorbar_cell(
    ax,
    block: dict,
    *,
    style: hw.HandwriteStyle,
    block_fs: float,
    text_color: str | None = None,
):
    """Vertical NLL color strip + numeric endpoints (matches voxel / CT heatmap)."""
    import numpy as np
    from matplotlib.patches import Rectangle

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)

    from tutorial_template import TutorialLayout

    inset = float(TutorialLayout().right_text_inset_frac)
    ax_w = max(float(ax.get_position().width), 1e-6)
    bar_x = inset / ax_w
    bar_w = float(block.get("colorbar_w_frac", CH4_NLL_COLORBAR_W_FRAC))
    bar_y0 = 0.06
    bar_y1 = 0.94
    n = 256
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    rgba = ch4_nll_heatmap_cmap()(t).reshape(n, 1, 4)
    ax.imshow(
        rgba,
        extent=(bar_x, bar_x + bar_w, bar_y0, bar_y1),
        aspect="auto",
        origin="lower",
        interpolation="bilinear",
        zorder=1,
    )
    ax.add_patch(
        Rectangle(
            (bar_x, bar_y0),
            bar_w,
            bar_y1 - bar_y0,
            transform=ax.transAxes,
            fill=False,
            edgecolor="#666666",
            linewidth=0.9,
            zorder=2,
        )
    )

    fs = float(block.get("block_fs", block_fs)) * 0.88
    fp = hw.hand_font(fs) if style.enabled else None
    label_x = bar_x + bar_w + 0.04
    ax.text(
        label_x,
        bar_y1,
        str(block.get("label_hi", _ch4_nll_heatmap_legend_fmt(block["nll_hi"]))),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fs,
        color=CH4_NLL_HEATMAP_COLORS[-1],
        fontproperties=fp,
    )
    ax.text(
        label_x,
        bar_y0,
        str(block.get("label_lo", _ch4_nll_heatmap_legend_fmt(block["nll_lo"]))),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fs,
        color=CH4_NLL_HEATMAP_COLORS[2],
        fontproperties=fp,
    )


def ch4_nll_viridis_color(nll, vmin, vmax, *, cmap_name: str = "viridis") -> str:
    del cmap_name  # kept for call-site compatibility
    return ch4_nll_heatmap_color(nll, vmin, vmax)


CH4_STORY_MATH_FS = 14.5
CH4_FORMULA_MATH_FS = round(CH4_STORY_MATH_FS * 1.4, 1)  # +40%
CH4_NOTATION_MATH_FS = round(CH4_STORY_MATH_FS * 1.3, 1)  # +30%
CH4_NOTATION_XI_MATH_FS = round(CH4_NOTATION_MATH_FS * 1.22, 1)
CH4_HERE_BLOCK_FS = 22.5
CH4_FORMULA_BODY_DROP_PT = 13.0 * 72.0 / 25.4
CH4_FORMULA_NLL_DROP_MM = 9.0    # raised 2 mm from prior 11 mm
CH4_FORMULA_NLL_DROP_PT = CH4_FORMULA_NLL_DROP_MM * 72.0 / 25.4
CH4_FORMULA_NLL_X_SHIFT_MM = -3.0   # single-column NLL (ch4_03 end, ch4_04+)
CH4_FORMULA_NLL_X_SHIFT_PT = CH4_FORMULA_NLL_X_SHIFT_MM * 72.0 / 25.4
CH4_FORMULA_PROB_DROP_PT = CH4_FORMULA_BODY_DROP_PT + 2.0 * 72.0 / 25.4   # +2.0 cm (ch4_04)
CH4_FORMULA_PROB_TEXT_X_FRAC = 0.02
CH4_FORMULA_PRIMARY_X_FRAC = 0.07
CH4_FORMULA_LOG_TEXT_X_FRAC = 0.015
CH4_FORMULA_CH03_LOG_NLL_X_SHIFT_PT = -20.0 * 72.0 / 25.4   # ch4_03 three-col: log + NLL left 2 cm
CH4_FORMULA_PROB_X_SHIFT_PT = 30.0 * 72.0 / 25.4             # prob right 3 cm
CH4_FORMULA_PROB_Y_SHIFT_PT = 12.0 * 72.0 / 25.4             # prob down 12 mm
CH4_FORMULA_LOG_LINE_DY_PT = 22.0
CH4_LOG_LIK_SPLIT_FRAC = 0.58

# Notation layout (expanded block: weights×3, y_i cases, x_i)
CH4_NOTATION_YI_LINE_IDX = 3
CH4_NOTATION_XI_LINE_IDX = 4
CH4_NOTATION_EXPANDED_TOP_PAD_MM = 1.5
CH4_NOTATION_EXPANDED_TOP_PAD_PT = CH4_NOTATION_EXPANDED_TOP_PAD_MM * 72.0 / 25.4
CH4_NOTATION_YI_DROP_MM = 6.0
CH4_NOTATION_YI_DROP_PT = CH4_NOTATION_YI_DROP_MM * 72.0 / 25.4
CH4_NOTATION_YI_XI_GAP_MM = 11.0
CH4_NOTATION_YI_XI_GAP_PT = CH4_NOTATION_YI_XI_GAP_MM * 72.0 / 25.4
CH4_NOTATION_XI_BODY_FS = round(CH4_HERE_BLOCK_FS * 26.5 / 21.5, 1)
CH4_NOTATION_XI_SUB_SCALE = round(CH4_HERE_BLOCK_FS * 0.62 / CH4_NOTATION_XI_BODY_FS, 4)

# TeX sources → Patrick Hand via ``latex_line_to_handwrite`` (limits, cases, scripts).
CH4_LIKELIHOOD_TEX = r"$\mathcal{L}=\prod_{i=0}^{N} p(y_i \mid x_i)$"
CH4_LOG_LIK_TEX = (
    r"$\log \mathcal{L}=\log(\prod_{i=0}^{N} p(y_i \mid x_i))"
    r" = \sum_{i=0}^{N} \log p(y_i \mid x_i)$"
)
CH4_LOG_LIK_LINE1_TEX = CH4_LOG_LIK_TEX
CH4_LOG_LIK_LINE2_TEX = ""
CH4_LOG_LIK_LOG_BOTH_TEX = CH4_LOG_LIK_TEX
CH4_LOG_LIK_SIMPLIFIED_TEX = CH4_LOG_LIK_TEX
CH4_NEG_LOG_LIK_TEX = r"$NLL(w_{\mathrm{ST}}, w_{\mathrm{EL}}, b)=-\log \mathcal{L}$"
CH4_NLL_FORMULA_TEX = (
    r"$NLL(w_{\mathrm{ST}}, w_{\mathrm{EL}}, b) = -\mathop{\sum}\limits_{i=0}^{N} \log p(y_i \mid x_i)$"
)
CH4_PROB_FORMULA_TEX = (
    r"$p(y_i \mid x_i) = \begin{cases}"
    r"\sigma(\boldsymbol{w_{\mathrm{ST}}} x_{i,\mathrm{ST}} + \boldsymbol{w_{\mathrm{EL}}} x_{i,\mathrm{EL}} + \boldsymbol{b})"
    r" & \text{if i passed} \\[1.0em]"
    r"1 - \sigma(\boldsymbol{w_{\mathrm{ST}}} x_{i,\mathrm{ST}} + \boldsymbol{w_{\mathrm{EL}}} x_{i,\mathrm{EL}} + \boldsymbol{b})"
    r" & \text{if i failed} \end{cases}$"
)
CH4_FORMULA_GRAD_UNIT_SHIFT_IN = 3.0 / 2.54   # shift ∂ column right (+1 cm)
CH4_FORMULA_GRAD_UNIT_BASE_DROP_MM = 12.0
CH4_FORMULA_GRAD_UNIT_LIFT_MM = 13.0          # lower on page → larger effective drop (+2 mm vs prior)
CH4_FORMULA_GRAD_UNIT_DROP_MM = CH4_FORMULA_GRAD_UNIT_BASE_DROP_MM - CH4_FORMULA_GRAD_UNIT_LIFT_MM
CH4_FORMULA_GRAD_UNIT_DROP_PT = CH4_FORMULA_GRAD_UNIT_DROP_MM * 72.0 / 25.4
CH4_FORMULA_GRAD_LINE_DY_MM = 3.5             # spacing within ∂ / update columns
CH4_FORMULA_GRAD_LINE_DY_PT = CH4_FORMULA_GRAD_LINE_DY_MM * 72.0 / 25.4
CH4_FORMULA_GRAD_ROW_PITCH_MM = 14.5
CH4_FORMULA_GRAD_ROW_PITCH_PT = CH4_FORMULA_GRAD_ROW_PITCH_MM * 72.0 / 25.4
CH4_FORMULA_GRAD_ROW_Y_PT = {
    0: 0.0,
    1: CH4_FORMULA_GRAD_ROW_PITCH_PT,
    2: 2.0 * CH4_FORMULA_GRAD_ROW_PITCH_PT,
}
CH4_FORMULA_GRAD_MATH_FS = round(CH4_FORMULA_MATH_FS * 0.88, 1)
CH4_GRAD_PARTIAL_ST_TEX = (
    r"$\frac{\partial NLL}{\partial w_{ST}}=\sum_i (p(y_i \mid x_i)-y_i) x_{i,ST}$"
)
CH4_GRAD_PARTIAL_EL_TEX = (
    r"$\frac{\partial NLL}{\partial w_{EL}}=\sum_i (p(y_i \mid x_i)-y_i) x_{i,EL}$"
)
CH4_GRAD_PARTIAL_B_TEX = (
    r"$\frac{\partial NLL}{\partial b}=\sum_i (p(y_i \mid x_i)-y_i)$"
)
CH4_GRAD_UPDATE_ST_TEX = (
    r"$w_{ST} \leftarrow w_{ST} - \alpha \frac{\partial NLL}{\partial w_{ST}}$"
)
CH4_GRAD_UPDATE_EL_TEX = (
    r"$w_{EL} \leftarrow w_{EL} - \alpha \frac{\partial NLL}{\partial w_{EL}}$"
)
CH4_GRAD_UPDATE_B_TEX = (
    r"$b \leftarrow b - \alpha \frac{\partial NLL}{\partial b}$"
)
CH4_GD_ARROW_ST_COLOR = "#2563eb"
CH4_GD_ARROW_EL_COLOR = "#ea580c"
CH4_GD_ARROW_B_COLOR = "#16a34a"
CH4_GD_GRADIENT_COLOR = "#d62728"
CH4_GD_PARTIAL_COLORS = (
    CH4_GD_ARROW_ST_COLOR,
    CH4_GD_ARROW_EL_COLOR,
    CH4_GD_ARROW_B_COLOR,
)
CH4_NOTATION_TEXT_DROP_MM = 1.5
CH4_NOTATION_TEXT_DROP_PT = CH4_NOTATION_TEXT_DROP_MM * 72.0 / 25.4
CH4_NOTATION_CORNER_YI_DROP_PT = 0.0
CH4_NOTATION_CORNER_YI_XI_GAP_MM = 6.0
CH4_NOTATION_CORNER_YI_XI_GAP_PT = CH4_NOTATION_CORNER_YI_XI_GAP_MM * 72.0 / 25.4
CH4_NOTATION_XI_LIFT_MM = 5.5
CH4_NOTATION_XI_LIFT_PT = -CH4_NOTATION_XI_LIFT_MM * 72.0 / 25.4
CH4_HERE_LOOSE_LINE_DY_PT = 5.0
CH4_HERE_WEIGHTS_LINE_DY_MM = 3.5
CH4_HERE_WEIGHTS_LINE_DY_PT = CH4_HERE_WEIGHTS_LINE_DY_MM * 72.0 / 25.4
CH4_HERE_PARTIALS_LINE_DY_MM = 8.0
CH4_HERE_PARTIALS_LINE_DY_PT = CH4_HERE_PARTIALS_LINE_DY_MM * 72.0 / 25.4
CH4_HERE_ALPHA_PRE_GAP_MM = 4.0
CH4_HERE_ALPHA_PRE_GAP_PT = CH4_HERE_ALPHA_PRE_GAP_MM * 72.0 / 25.4
CH4_HERE_NLL_TOP_PAD_MM = -4.0   # pull NLL text up within its row
CH4_HERE_NLL_TOP_PAD_PT = CH4_HERE_NLL_TOP_PAD_MM * 72.0 / 25.4
CH4_HERE_NLL_PRE_GAP_MM = -4.0   # tighten row gap between bias and NLL
CH4_HERE_NLL_PRE_GAP_PT = CH4_HERE_NLL_PRE_GAP_MM * 72.0 / 25.4
CH4_HERE_NLL_EQUALS_EXTRA_GAP_MM = 2.0
CH4_HERE_NLL_EQUALS_EXTRA_GAP_PT = CH4_HERE_NLL_EQUALS_EXTRA_GAP_MM * 72.0 / 25.4
CH4_CASES_ROW_GAP_MM = 10.0
CH4_CASES_ROW_GAP_PT = CH4_CASES_ROW_GAP_MM * 72.0 / 25.4


def _ch4_fmt_step_size(v) -> str:
    return f"{float(v):.4f}".rstrip("0").rstrip(".")


def ch4_we_are_here_grad_line_colors(*, grad_red: bool = False, transition_u: float = 0.0) -> list[str]:
    """Per-partial colors for the right rail (matches GD axis-arrow hues; blends to red in ch4_07)."""
    import matplotlib.colors as mcolors
    import numpy as np

    red = CH4_GD_GRADIENT_COLOR
    u = float(np.clip(float(transition_u), 0.0, 1.0))
    if u > 0.0:
        out: list[str] = []
        for c in CH4_GD_PARTIAL_COLORS:
            a = np.array(mcolors.to_rgb(c))
            b = np.array(mcolors.to_rgb(red))
            out.append(mcolors.to_hex(a + u * (b - a)))
        return out
    if grad_red:
        return [red, red, red]
    return list(CH4_GD_PARTIAL_COLORS)


def ch4_write_slot_count(blocks, *, style=None) -> int:
    """Count handwriting reveal slots across blocks (labels + lines)."""
    import handwrite_tutorial as hw

    style = style or CH4_COMPOSER.handwrite_style()
    n = 0
    for block in blocks or []:
        if block.get("label"):
            n += 1
        n += hw.block_n_lines(block, style=style)
    return n


def ch4_blocks_write_from_slot(
    blocks,
    start_slot: int,
    progress: float,
    *,
    style=None,
) -> dict[int, dict]:
    """Per-block line progress keeping the first ``start_slot`` slots fully written."""
    return ch4_group_write_from_slot([list(blocks or [])], start_slot, progress, style=style)[0]


def ch4_bold_mathtext(tex: str) -> str:
    s = str(tex).strip()
    if s.startswith("$") and s.endswith("$"):
        inner = s[1:-1]
        return rf"${{\boldmath {inner}}}$"
    return rf"${{\boldmath {s}}}$"


def ch4_we_are_here_blocks(
    w_st,
    w_el,
    b,
    nll,
    *,
    nll_vmin=None,
    nll_vmax=None,
    point_color=None,
    grad=None,
    step_size=None,
    grad_line_colors=None,
) -> list[dict]:
    """Right-rail state at the current 3-D point (single-line rows, tight spacing)."""
    row = {
        "block_fs": CH4_HERE_BLOCK_FS,
        "line_dy_pt": CH4_HERE_LOOSE_LINE_DY_PT,
        "top_pad_pt": 0.0,
        "label_gap_pt": 0.0,
        "align": "left",
        "pt_units": True,
    }
    weights_row = {**row, "line_dy_pt": CH4_HERE_WEIGHTS_LINE_DY_PT}
    partials_row = {**row, "line_dy_pt": CH4_HERE_PARTIALS_LINE_DY_PT}
    blocks = [{
        **weights_row,
        "text": (
            f"w_ST = {float(w_st):.2f}\n"
            f"w_EL = {float(w_el):.2f}\n"
            f"b = {float(b):.2f}"
        ),
        "role": "weights",
    }]
    nll_color = None
    if nll_vmin is not None and nll_vmax is not None:
        nll_color = ch4_nll_viridis_color(nll, nll_vmin, nll_vmax)
    elif point_color is not None:
        nll_color = str(point_color)
    nll_block = {
        **row,
        "text": (
            f"NLL({float(w_st):.2f}, {float(w_el):.2f}, {float(b):.2f})\n"
            f"= {float(nll):.2f}"
        ),
        "top_pad_pt": CH4_HERE_NLL_TOP_PAD_PT,
        "pre_gap_pt": CH4_HERE_NLL_PRE_GAP_PT,
        "line_dy_pt": CH4_HERE_WEIGHTS_LINE_DY_PT,
        "line_extra_dy_pt": {0: CH4_HERE_NLL_EQUALS_EXTRA_GAP_PT},
        "role": "nll",
    }
    if nll_color:
        nll_block["text_color"] = nll_color
    blocks.append(nll_block)
    if grad is not None:
        g1, g2, gb = grad
        colors = list(grad_line_colors) if grad_line_colors is not None else list(CH4_GD_PARTIAL_COLORS)
        grad_block = {
            **partials_row,
            "text": (
                rf"$\partial NLL/\partial w_{{ST}} = {float(g1):.2f}$" + "\n"
                rf"$\partial NLL/\partial w_{{EL}} = {float(g2):.2f}$" + "\n"
                rf"$\partial NLL/\partial b = {float(gb):.2f}$"
            ),
            "role": "gradient",
        }
        if colors:
            grad_block["line_text_colors"] = list(colors[:3])
        blocks.append(grad_block)
    if step_size is not None:
        blocks.append({
            **row,
            "text": f"α = {_ch4_fmt_step_size(step_size)}",
            "pre_gap_pt": CH4_HERE_ALPHA_PRE_GAP_PT,
            "role": "gradient",
        })
    return blocks


def ch4_we_are_here_roc_blocks(
    w_st,
    w_el,
    b,
    nll,
    rocs,
    *,
    nll_vmin=None,
    nll_vmax=None,
    point_color=None,
    show_partials: bool = False,
    partials=None,
    grad_line_colors=None,
) -> list[dict]:
    """Right-rail: weights + NLL + average rate-of-change (or ∂ lines when ``show_partials``)."""
    row = {
        "block_fs": CH4_HERE_BLOCK_FS,
        "line_dy_pt": CH4_HERE_LOOSE_LINE_DY_PT,
        "top_pad_pt": 0.0,
        "label_gap_pt": 0.0,
        "align": "left",
        "pt_units": True,
    }
    weights_row = {**row, "line_dy_pt": CH4_HERE_WEIGHTS_LINE_DY_PT}
    partials_row = {**row, "line_dy_pt": CH4_HERE_PARTIALS_LINE_DY_PT}
    blocks = [{
        **weights_row,
        "text": (
            f"w_ST = {float(w_st):.2f}\n"
            f"w_EL = {float(w_el):.2f}\n"
            f"b = {float(b):.2f}"
        ),
        "role": "weights",
    }]
    nll_color = None
    if nll_vmin is not None and nll_vmax is not None:
        nll_color = ch4_nll_viridis_color(nll, nll_vmin, nll_vmax)
    elif point_color is not None:
        nll_color = str(point_color)
    nll_block = {
        **row,
        "text": (
            f"NLL({float(w_st):.2f}, {float(w_el):.2f}, {float(b):.2f})\n"
            f"= {float(nll):.2f}"
        ),
        "top_pad_pt": CH4_HERE_NLL_TOP_PAD_PT,
        "pre_gap_pt": CH4_HERE_NLL_PRE_GAP_PT,
        "line_dy_pt": CH4_HERE_WEIGHTS_LINE_DY_PT,
        "line_extra_dy_pt": {0: CH4_HERE_NLL_EQUALS_EXTRA_GAP_PT},
        "role": "nll",
    }
    if nll_color:
        nll_block["text_color"] = nll_color
    blocks.append(nll_block)
    if rocs is not None:
        g1, g2, gb = rocs
        colors = list(grad_line_colors) if grad_line_colors is not None else list(CH4_GD_PARTIAL_COLORS)
        if show_partials and partials is not None:
            p1, p2, pb = partials
            roc_block = {
                **partials_row,
                "text": (
                    rf"$\partial NLL/\partial w_{{ST}} = {float(p1):.2f}$" + "\n"
                    rf"$\partial NLL/\partial w_{{EL}} = {float(p2):.2f}$" + "\n"
                    rf"$\partial NLL/\partial b = {float(pb):.2f}$"
                ),
                "role": "gradient",
            }
        else:
            roc_block = {
                **partials_row,
                "text": (
                    rf"$\Delta NLL/\Delta w_{{ST}} = {float(g1):.2f}$" + "\n"
                    rf"$\Delta NLL/\Delta w_{{EL}} = {float(g2):.2f}$" + "\n"
                    rf"$\Delta NLL/\Delta b = {float(gb):.2f}$"
                ),
                "role": "roc",
            }
        if colors:
            roc_block["line_text_colors"] = list(colors[:3])
        blocks.append(roc_block)
    return blocks


def ch4_we_are_here_ct_blocks(
    sweep_axis,
    plane_val,
    bounds,
    *,
    show_sweep_range: bool = False,
) -> list[dict]:
    """Right-rail for CT slice animation — plane value or sweep range per param."""
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = (float(v) for v in bounds)
    row = {
        "block_fs": CH4_HERE_BLOCK_FS,
        "line_dy_pt": CH4_HERE_WEIGHTS_LINE_DY_PT,
        "top_pad_pt": 0.0,
        "label_gap_pt": 0.0,
        "align": "left",
        "pt_units": True,
        "role": "weights",
    }

    def _range(lo, hi, label):
        return f"{label} ∈ [{lo:.2f}, {hi:.2f}]"

    def _val(label, v):
        return f"{label} = {float(v):.2f}"

    axis = str(sweep_axis)
    if axis == "st":
        w_st = _range(dlo1, dhi1, "w_ST") if show_sweep_range else _val("w_ST", plane_val)
        w_el = _range(dlo2, dhi2, "w_EL")
        b_line = _range(dlob, dhib, "b")
    elif axis == "el":
        w_st = _range(dlo1, dhi1, "w_ST")
        w_el = _range(dlo2, dhi2, "w_EL") if show_sweep_range else _val("w_EL", plane_val)
        b_line = _range(dlob, dhib, "b")
    else:
        w_st = _range(dlo1, dhi1, "w_ST")
        w_el = _range(dlo2, dhi2, "w_EL")
        b_line = _range(dlob, dhib, "b") if show_sweep_range else _val("b", plane_val)
    return [{**row, "text": f"{w_st}\n{w_el}\n{b_line}"}]


def ch4_measurements_blocks(w_st, w_el, b, nll, *, study=None, exam=None, y=None, **kw):
    del study, exam, y
    return ch4_we_are_here_blocks(w_st, w_el, b, nll, **kw)


def _ch4_story_mathtext_block(*lines: str, **kw) -> dict:
    base = {
        "mathtext_usetex": True,
        "mathtext_fs": CH4_STORY_MATH_FS,
        "align": "center",
        "role": "formula",
        "pt_units": True,
    }
    base.update(kw)
    base["mathtext_lines"] = list(lines)
    return base


def _ch4_formula_hand_block(text: str, **kw) -> dict:
    """Handwritten formula block (Patrick Hand, same as knob notation)."""
    base = dict(
        text=str(text),
        block_fs=CH4_HERE_BLOCK_FS,
        top_pad_pt=0.0,
        text_y_inset_pt=CH4_FORMULA_NLL_DROP_PT,
        text_x_frac=CH4_FORMULA_PRIMARY_X_FRAC,
        align="left",
        pt_units=True,
        role="formula",
        bold_lhs=True,
    )
    base.update(kw)
    return base


def ch4_formula_primary_block(tex: str, **kw) -> dict:
    """Primary bottom formula — handwritten."""
    base = dict(weight=0.30, formula_slot="nll")
    base.update(kw)
    return _ch4_formula_hand_block(tex, **base)


def ch4_formula_col_likelihood(**kw) -> dict:
    base = dict(formula_slot="lik", weight=0.26, text_x_frac=0.05, bold_lhs=True)
    base.update(kw)
    return _ch4_formula_hand_block(CH4_LIKELIHOOD_TEX, **base)


def ch4_formula_col_log(**kw) -> dict:
    base = dict(
        formula_slot="log",
        weight=0.44,
        text_x_frac=CH4_FORMULA_LOG_TEXT_X_FRAC,
        text_x_shift_pt=CH4_FORMULA_CH03_LOG_NLL_X_SHIFT_PT,
        bold_lhs=False,
    )
    base.update(kw)
    return _ch4_formula_hand_block(CH4_LOG_LIK_TEX, **base)


def ch4_formula_col_nll(**kw) -> dict:
    base = dict(
        formula_slot="nll_col",
        weight=0.28,
        text_x_frac=0.05,
        text_x_shift_pt=CH4_FORMULA_CH03_LOG_NLL_X_SHIFT_PT,
        bold_lhs=True,
    )
    base.update(kw)
    return _ch4_formula_hand_block(CH4_NLL_FORMULA_TEX, **base)


def ch4_formula_blocks_ch4_03_three_col() -> list[dict]:
    """ch4_03 bottom row: likelihood | log-likelihood (single line) | NLL."""
    return [
        ch4_formula_col_likelihood(),
        ch4_formula_col_log(),
        ch4_formula_col_nll(),
    ]


def ch4_bottom_prog_ch4_03_three_col(
    *,
    lik_u: float = 1.0,
    log_line_us: tuple[float, float] = (0.0, 0.0),
    nll_u: float = 0.0,
) -> dict[int, dict]:
    """Per-block line progress for the ch4_03 three-column formula row."""
    l0, l1 = (float(log_line_us[0]), float(log_line_us[1]))
    split = float(CH4_LOG_LIK_SPLIT_FRAC)
    if l1 <= 0.0:
        log_prog = l0 * split
    else:
        log_prog = split + l1 * (1.0 - split)
    return {
        0: {0: float(lik_u)},
        1: {0: float(log_prog)},
        2: {0: float(nll_u)},
    }


def ch4_formula_nll_block(**kw) -> dict:
    """Shared NLL formula block (ch4_03 end + ch4_04–06 left column)."""
    base = dict(
        formula_slot="nll",
        weight=0.30,
        text_x_shift_pt=CH4_FORMULA_NLL_X_SHIFT_PT,
    )
    base.update(kw)
    return _ch4_formula_hand_block(CH4_NLL_FORMULA_TEX, **base)


def ch4_formula_prob_block(**kw) -> dict:
    """Piecewise p(y_i | x_i) with cases (ch4_03 end + ch4_04 bottom)."""
    base = dict(
        text_y_inset_pt=-40.0 + CH4_FORMULA_PROB_DROP_PT,
        text_y_shift_pt=CH4_FORMULA_PROB_Y_SHIFT_PT,
        text_x_frac=CH4_FORMULA_PROB_TEXT_X_FRAC,
        text_x_shift_pt=CH4_FORMULA_PROB_X_SHIFT_PT,
        weight=0.70,
        formula_slot="prob",
        bold_lhs=False,
        line_dy_pt=CH4_FORMULA_LOG_LINE_DY_PT,
        cases_row_gap_pt=CH4_CASES_ROW_GAP_PT,
    )
    base.update(kw)
    return _ch4_formula_hand_block(CH4_PROB_FORMULA_TEX, **base)


def ch4_formula_blocks_nll_story() -> list[dict]:
    return [ch4_formula_nll_block()]


def ch4_formula_grad_unit_block(**kw) -> dict:
    """Three ∂NLL formulas stacked as one movable unit (ch4_06)."""
    partials = "\n".join([CH4_GRAD_PARTIAL_ST_TEX, CH4_GRAD_PARTIAL_EL_TEX, CH4_GRAD_PARTIAL_B_TEX])
    base = dict(
        top_pad_pt=0.0,
        text_y_inset_pt=CH4_FORMULA_GRAD_UNIT_DROP_PT,
        text_x_frac=0.04,
        line_dy_pt=CH4_FORMULA_GRAD_LINE_DY_PT,
        line_row_y_pt=CH4_FORMULA_GRAD_ROW_Y_PT,
        formula_slot="grad",
        formula_grad_shift_in=CH4_FORMULA_GRAD_UNIT_SHIFT_IN,
        weight=0.36,
        bold_lhs=False,
    )
    base.update(kw)
    return _ch4_formula_hand_block(partials, **base)


def ch4_formula_update_block(**kw) -> dict:
    mlines = kw.pop("mathtext_lines", None)
    text = kw.pop("text", None)
    if text is None:
        lines = mlines if mlines is not None else [
            CH4_GRAD_UPDATE_ST_TEX,
            CH4_GRAD_UPDATE_EL_TEX,
            CH4_GRAD_UPDATE_B_TEX,
        ]
        text = "\n".join(str(ln) for ln in lines)
    base = dict(
        top_pad_pt=0.0,
        text_y_inset_pt=CH4_FORMULA_GRAD_UNIT_DROP_PT,
        text_x_frac=0.02,
        line_dy_pt=CH4_FORMULA_GRAD_LINE_DY_PT,
        line_row_y_pt=CH4_FORMULA_GRAD_ROW_Y_PT,
        formula_slot="update",
        formula_grad_shift_in=CH4_FORMULA_GRAD_UNIT_SHIFT_IN,
        weight=0.34,
        bold_lhs=False,
    )
    base.update(kw)
    return _ch4_formula_hand_block(text, **base)


def ch4_notation_corner_blocks() -> list[dict]:
    return [_ch4_notation_yi_xi_block(
        text_y_inset_pt=CH4_NOTATION_TEXT_DROP_PT,
        line_y_inset_pt={0: CH4_NOTATION_CORNER_YI_DROP_PT},
        line_extra_dy_pt={0: CH4_NOTATION_CORNER_YI_XI_GAP_PT},
        line_body_fs={1: CH4_NOTATION_XI_BODY_FS},
        line_subscript_scale={1: CH4_NOTATION_XI_SUB_SCALE},
    )]


def ch4_formula_blocks_3d_story() -> list[dict]:
    return [
        ch4_formula_nll_block(),
        ch4_formula_prob_block(),
    ]


def _ch4_bold_handwrite_line(tex: str) -> str:
    """Wrap a ``$...$`` formula line for Patrick Hand bold (``<<B>>`` markers)."""
    from handwrite_tutorial import BOLD_CLOSE, BOLD_OPEN

    s = str(tex).strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    return f"{BOLD_OPEN}{s}{BOLD_CLOSE}"


def ch4_formula_blocks_gd_story(
    *,
    bold_update_idx: int | None = None,
    bold_all_updates: bool = False,
    grad_red: bool = False,
) -> list[dict]:
    updates = [CH4_GRAD_UPDATE_ST_TEX, CH4_GRAD_UPDATE_EL_TEX, CH4_GRAD_UPDATE_B_TEX]
    if bold_all_updates:
        updates = [_ch4_bold_handwrite_line(u) for u in updates]
    elif bold_update_idx is not None:
        i = int(bold_update_idx)
        if 0 <= i < len(updates):
            updates = list(updates)
            updates[i] = _ch4_bold_handwrite_line(updates[i])
    return [
        ch4_formula_nll_block(),
        ch4_formula_grad_unit_block(),
        ch4_formula_update_block(text="\n".join(updates)),
    ]


def ch4_cached_notation_corner_blocks() -> list[dict]:
    global _CH4_CACHED_NOTATION_CORNER
    if _CH4_CACHED_NOTATION_CORNER is None:
        _CH4_CACHED_NOTATION_CORNER = ch4_notation_corner_blocks()
    return _CH4_CACHED_NOTATION_CORNER


def ch4_cached_formula_blocks_3d_story() -> list[dict]:
    global _CH4_CACHED_3D_BOTTOM
    if _CH4_CACHED_3D_BOTTOM is None:
        _CH4_CACHED_3D_BOTTOM = ch4_formula_blocks_3d_story()
    return _CH4_CACHED_3D_BOTTOM


def ch4_cached_formula_blocks_gd_story() -> list[dict]:
    global _CH4_CACHED_GD_BOTTOM
    if _CH4_CACHED_GD_BOTTOM is None:
        _CH4_CACHED_GD_BOTTOM = ch4_formula_blocks_gd_story()
    return _CH4_CACHED_GD_BOTTOM


def ch4_rails_cache_key(*, gd_formulas: bool = False) -> str | None:
    """Cache key for static bottom/corner rails when layout is fully revealed."""
    return CH4_RAILS_CACHE_GD if gd_formulas else CH4_RAILS_CACHE_3D


def ch4_notation_blocks_condensed() -> list[dict]:
    return ch4_cached_notation_corner_blocks()


def ch4_bottom_formulas_nll_with_notation_corner(**_ignored):
    """Deprecated: use ``ch4_formula_blocks_3d_story()`` + ``ch4_notation_corner_blocks()``."""
    return ch4_formula_blocks_3d_story()


def ch4_compose_tutorial_frame(
    plot_img,
    *,
    math_right_blocks=None,
    math_bottom_blocks=None,
    right_blocks=None,
    bottom_blocks=None,
    corner_blocks=None,
    write_progress=1.0,
    layout_u=1.0,
    panel_u=1.0,
    plot_start_rect=None,
    right_write_progress=None,
    bottom_write_progress=None,
    progress_override=None,
    plot_alpha=1.0,
    title_write_progress=None,
    theme: str | None = None,
    composer: TutorialComposer | None = None,
    right_section_title=CH4_RIGHT_SECTION_TITLE,
    bottom_section_title=CH4_BOTTOM_SECTION_TITLE,
    right_title=None,
    bottom_title=None,
    corner_title=None,
    right_title_color=None,
    right_title_single_line=False,
    rails_cache_key=None,
    shell_cache_key=None,
    **_ignored,
):
    comp = composer or (make_composer(theme) if theme else CH4_COMPOSER)
    scene = TutorialScene(
        plot=plot_img,
        math_right_blocks=right_blocks if right_blocks is not None else math_right_blocks,
        math_bottom_blocks=bottom_blocks if bottom_blocks is not None else math_bottom_blocks,
        math_corner_blocks=corner_blocks,
        right_section_title=right_title if right_title is not None else right_section_title,
        bottom_section_title=bottom_title if bottom_title is not None else bottom_section_title,
        corner_section_title=corner_title if corner_title is not None else CH4_NOTATION_SECTION_TITLE,
        right_title_color=right_title_color,
        right_title_single_line=bool(right_title_single_line),
    )
    return comp.compose_frame(
        scene,
        write_progress=write_progress,
        layout_u=layout_u,
        panel_u=panel_u,
        plot_start_rect=plot_start_rect,
        right_write_progress=right_write_progress,
        bottom_write_progress=bottom_write_progress,
        progress_override=progress_override,
        plot_alpha=plot_alpha,
        title_write_progress=title_write_progress,
        rails_cache_key=rails_cache_key,
        shell_cache_key=shell_cache_key,
    )


def ch4_bottom_per_block_progress(
    blocks: list[dict],
    progress_by_idx: dict[int, float],
    *,
    style=None,
) -> dict[int, dict]:
    """Per-block write/erase progress for bottom formula columns."""
    import handwrite_tutorial as hw

    style = style or CH4_COMPOSER.handwrite_style()
    out: dict[int, dict] = {}
    for bi, block in enumerate(blocks):
        p = float(progress_by_idx.get(bi, 1.0))
        bp = hw.block_write_progress([block], p, style=style)
        if 0 in bp:
            out[bi] = bp[0]
    return out


def ch4_group_write_from_slot(block_groups, start_slot: int, progress: float, *, style=None):
    """Stagger reveal across groups, keeping the first ``start_slot`` slots fully written."""
    import handwrite_tutorial as hw

    style = style or CH4_COMPOSER.handwrite_style()
    slots = []
    for gi, blocks in enumerate(block_groups):
        for bi, block in enumerate(blocks):
            if block.get("label"):
                slots.append((gi, bi, "label", None))
            for li in range(hw.block_n_lines(block, style=style)):
                slots.append((gi, bi, "line", li))
    if not slots:
        return [{} for _ in block_groups]
    start_slot = int(max(0, min(start_slot, len(slots))))
    tail_n = max(len(slots) - start_slot, 1)
    tail_prog = hw.line_write_progresses(tail_n, float(progress), overlap=style.reveal_overlap)
    out: list[dict[int, dict]] = [{} for _ in block_groups]
    for i, (gi, bi, kind, li) in enumerate(slots):
        p = 1.0 if i < start_slot else float(tail_prog[i - start_slot])
        out[gi].setdefault(bi, {})
        if kind == "label":
            out[gi][bi]["__label__"] = p
        else:
            out[gi][bi][int(li)] = p
    return out


def ch4_blend_images(img_a, img_b, u: float):
    import numpy as np
    from PIL import Image

    u = float(np.clip(u, 0.0, 1.0))
    a = np.asarray(img_a.convert("RGBA"), dtype=np.float32)
    b = np.asarray(img_b.convert("RGBA"), dtype=np.float32)
    if a.shape != b.shape:
        b = np.asarray(img_b.convert("RGBA").resize(img_a.size, Image.Resampling.LANCZOS), dtype=np.float32)
    out = (1.0 - u) * a + u * b
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGBA").convert("RGB")


def ch4_crossfade_frames(img_a, img_b, u: float):
    return ch4_blend_images(img_a, img_b, u)


compose_tutorial = ch4_compose_tutorial_frame
crossfade_to_tutorial = ch4_crossfade_frames
blend_images = ch4_blend_images


def ch4_save_png(img, path):
    path = Path(path)
    img.save(path, format="PNG")
    print(f"wrote {path}  ({img.size[0]}×{img.size[1]})")


def ch4_scene_from_dict(d: dict) -> TutorialScene:
    return TutorialScene(
        plot=d["plot"],
        math_right_blocks=d.get("math_right_blocks"),
        math_bottom_blocks=d.get("math_bottom_blocks"),
        right_section_title=d.get("right_section_title", CH4_RIGHT_SECTION_TITLE),
        bottom_section_title=d.get("bottom_section_title", CH4_BOTTOM_SECTION_TITLE),
    )


def ch4_render_tutorial_frame(scene_dict, write_progress=1.0, *, theme: str | None = None, composer=None):
    comp = composer or (make_composer(theme) if theme else CH4_COMPOSER)
    return comp.render_scene(ch4_scene_from_dict(scene_dict), write_progress=write_progress)


def ch4_export_theme_demos(
    scene_dict,
    *,
    themes=None,
    n_frames=32,
    ms_per_frame=100,
    save_mp4,
    prefix="ch4_theme",
):
    """Export one handwrite demo MP4 per color theme."""
    paths = []
    for name in (themes or list_themes()):
        comp = make_composer(name)
        fn = f"{prefix}_{name}.mp4"
        scene = ch4_scene_from_dict(scene_dict)
        paths.append(comp.export_mp4(scene, fn, save_mp4=save_mp4, output_dir=OUTPUT_DIR, n_frames=n_frames, ms_per_frame=ms_per_frame))
    return paths


print("Chapter 4 layout OK — handwriting:", hw.active_handwriting_family())
