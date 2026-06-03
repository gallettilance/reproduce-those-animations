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

CH4_RAILS_CACHE_3D = "ch4_lik_3d_v1"
CH4_RAILS_CACHE_GD = "ch4_lik_gd_v1"

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
        r"$y_{i} = \begin{cases} 1 & \text{if student i passed} \\ 0 & \text{if student i failed} \end{cases}$",
        r"$x_{i} = \left[x_{i,\mathrm{ST}},\, x_{i,\mathrm{EL}}\right]$",
    )


def _ch4_notation_yi_xi_block(**extra) -> dict:
    yi, xi = _ch4_notation_yi_xi_lines()
    base = dict(
        mathtext_fs=[CH4_NOTATION_MATH_FS, CH4_NOTATION_XI_MATH_FS],
        mathtext_line_y_inset_pt=[0.0, CH4_NOTATION_XI_LIFT_PT],
        line_dy_pt=CH4_HERE_LOOSE_LINE_DY_PT,
        align="left",
        top_pad_pt=0.0,
        pt_units=True,
        role="formula",
        mathtext_usetex=True,
    )
    base.update(extra)
    return _ch4_story_mathtext_block(yi, xi, **base)


def ch4_notation_blocks_basic() -> list[dict]:
    return [_ch4_notation_weights_row()]


def ch4_notation_blocks_expanded() -> list[dict]:
    """Right-rail notation for ch4_03 — same typography as 04+ We are here."""
    yi, xi = _ch4_notation_yi_xi_lines()
    return [{
        "block_fs": CH4_HERE_BLOCK_FS,
        "line_dy_pt": CH4_HERE_WEIGHTS_LINE_DY_PT,
        "top_pad_pt": 0.0,
        "label_gap_pt": 0.0,
        "align": "left",
        "pt_units": True,
        "text": "w_ST = knob 1\nw_EL = knob 2\nb = knob 3",
        "role": "weights",
        "mathtext_usetex": True,
        "mathtext_fs": [CH4_NOTATION_MATH_FS, CH4_NOTATION_XI_MATH_FS],
        "mathtext_line_y_inset_pt": [CH4_HERE_LOOSE_LINE_DY_PT, CH4_NOTATION_XI_LIFT_PT],
        "mathtext_lines": [yi, xi],
    }]


def ch4_formula_blocks_likelihood() -> list[dict]:
    return [{"text": "ℒ = Π_i p(y_i | x_i)", "bold_lhs": True, "role": "formula"}]


def ch4_formula_blocks_log_likelihood() -> list[dict]:
    return [{"text": "log ℒ = Σ_i log p(y_i | x_i)", "bold_lhs": True, "role": "formula"}]


def ch4_formula_blocks_nll() -> list[dict]:
    return [{"text": "NLL = -Σ_i log p(y_i | x_i)", "bold_lhs": True, "role": "nll"}]


def ch4_nll_viridis_color(nll, vmin, vmax, *, cmap_name: str = "viridis") -> str:
    import matplotlib as mpl

    lo, hi = float(vmin), float(vmax)
    t = 0.0 if hi <= lo else float((float(nll) - lo) / (hi - lo))
    t = max(0.0, min(1.0, t))
    rgba = mpl.colormaps.get_cmap(cmap_name)(t)
    return mpl.colors.to_hex(rgba, keep_alpha=False)


CH4_STORY_MATH_FS = 14.5
CH4_FORMULA_MATH_FS = round(CH4_STORY_MATH_FS * 1.4, 1)  # +40%
CH4_NOTATION_MATH_FS = round(CH4_STORY_MATH_FS * 1.3, 1)  # +30%
CH4_NOTATION_XI_MATH_FS = round(CH4_NOTATION_MATH_FS * 1.22, 1)
CH4_HERE_BLOCK_FS = 21.5
CH4_FORMULA_BODY_DROP_PT = 13.0 * 72.0 / 25.4
CH4_FORMULA_NLL_DROP_MM = 9.0    # raised 2 mm from prior 11 mm
CH4_FORMULA_NLL_DROP_PT = CH4_FORMULA_NLL_DROP_MM * 72.0 / 25.4
CH4_FORMULA_PROB_DROP_PT = CH4_FORMULA_BODY_DROP_PT + 1.0 * 72.0 / 25.4   # +1.0 cm (ch4_04)
CH4_FORMULA_PROB_TEXT_X_FRAC = 0.02
CH4_FORMULA_GRAD_UNIT_SHIFT_IN = 3.0 / 2.54   # shift ∂ column right (+1 cm)
CH4_FORMULA_GRAD_UNIT_BASE_DROP_MM = 12.0
CH4_FORMULA_GRAD_UNIT_LIFT_MM = 13.0          # raise ∂ / update columns slightly
CH4_FORMULA_GRAD_UNIT_DROP_MM = CH4_FORMULA_GRAD_UNIT_BASE_DROP_MM - CH4_FORMULA_GRAD_UNIT_LIFT_MM
CH4_FORMULA_GRAD_UNIT_DROP_PT = CH4_FORMULA_GRAD_UNIT_DROP_MM * 72.0 / 25.4
CH4_FORMULA_GRAD_LINE_DY_MM = 2.8             # spacing within ∂ / update columns
CH4_FORMULA_GRAD_LINE_DY_PT = CH4_FORMULA_GRAD_LINE_DY_MM * 72.0 / 25.4
CH4_FORMULA_GRAD_MATH_FS = round(CH4_FORMULA_MATH_FS * 0.88, 1)
CH4_NLL_FORMULA_TEX = (
    r"$NLL = -\mathop{\sum}\limits_{i=0}^{N} \log p(y_i \mid x_i)$"
)
CH4_GRAD_PARTIAL_ST_TEX = (
    r"$\frac{\partial NLL}{\partial w_{\mathrm{ST}}} = "
    r"\sum_i \bigl(p(y_i \mid x_i) - y_i\bigr)\, x_{i,\mathrm{ST}}$"
)
CH4_GRAD_PARTIAL_EL_TEX = (
    r"$\frac{\partial NLL}{\partial w_{\mathrm{EL}}} = "
    r"\sum_i \bigl(p(y_i \mid x_i) - y_i\bigr)\, x_{i,\mathrm{EL}}$"
)
CH4_GRAD_PARTIAL_B_TEX = (
    r"$\frac{\partial NLL}{\partial b} = \sum_i \bigl(p(y_i \mid x_i) - y_i\bigr)$"
)
CH4_GRAD_UPDATE_ST_TEX = (
    r"$w_{\mathrm{ST}} \leftarrow w_{\mathrm{ST}}"
    r" - \alpha\,\frac{\partial NLL}{\partial w_{\mathrm{ST}}}$"
)
CH4_GRAD_UPDATE_EL_TEX = (
    r"$w_{\mathrm{EL}} \leftarrow w_{\mathrm{EL}}"
    r" - \alpha\,\frac{\partial NLL}{\partial w_{\mathrm{EL}}}$"
)
CH4_GRAD_UPDATE_B_TEX = (
    r"$b \leftarrow b - \alpha\,\frac{\partial NLL}{\partial b}$"
)
CH4_GD_ARROW_ST_COLOR = "#2563eb"
CH4_GD_ARROW_EL_COLOR = "#ea580c"
CH4_GD_ARROW_B_COLOR = "#16a34a"
CH4_GD_PARTIAL_COLORS = (
    CH4_GD_ARROW_ST_COLOR,
    CH4_GD_ARROW_EL_COLOR,
    CH4_GD_ARROW_B_COLOR,
)
CH4_NOTATION_TEXT_DROP_MM = 1.0
CH4_NOTATION_TEXT_DROP_PT = CH4_NOTATION_TEXT_DROP_MM * 72.0 / 25.4
CH4_NOTATION_XI_LIFT_MM = 3.0
CH4_NOTATION_XI_LIFT_PT = -CH4_NOTATION_XI_LIFT_MM * 72.0 / 25.4
CH4_HERE_LOOSE_LINE_DY_PT = 5.0
CH4_HERE_WEIGHTS_LINE_DY_MM = 3.5
CH4_HERE_WEIGHTS_LINE_DY_PT = CH4_HERE_WEIGHTS_LINE_DY_MM * 72.0 / 25.4
CH4_HERE_PARTIALS_LINE_DY_PT = CH4_HERE_LOOSE_LINE_DY_PT * 0.5
CH4_HERE_NLL_TOP_PAD_MM = -4.0   # pull NLL text up within its row
CH4_HERE_NLL_TOP_PAD_PT = CH4_HERE_NLL_TOP_PAD_MM * 72.0 / 25.4
CH4_HERE_NLL_PRE_GAP_MM = -4.0   # tighten row gap between bias and NLL
CH4_HERE_NLL_PRE_GAP_PT = CH4_HERE_NLL_PRE_GAP_MM * 72.0 / 25.4


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
        "text": f"NLL = {float(nll):.2f}",
        "top_pad_pt": CH4_HERE_NLL_TOP_PAD_PT,
        "pre_gap_pt": CH4_HERE_NLL_PRE_GAP_PT,
        "role": "nll",
    }
    if nll_color:
        nll_block["text_color"] = nll_color
    blocks.append(nll_block)
    if grad is not None:
        g1, g2, gb = grad
        colors = grad_line_colors if grad_line_colors is not None else CH4_GD_PARTIAL_COLORS
        blocks.append({
            **partials_row,
            "mathtext_lines": [
                rf"$\frac{{\partial NLL}}{{\partial w_{{\mathrm{{ST}}}}}} = {float(g1):.2f}$",
                rf"$\frac{{\partial NLL}}{{\partial w_{{\mathrm{{EL}}}}}} = {float(g2):.2f}$",
                rf"$\frac{{\partial NLL}}{{\partial b}} = {float(gb):.2f}$",
            ],
            "mathtext_line_colors": list(colors),
            "mathtext_usetex": True,
            "mathtext_fs": CH4_HERE_BLOCK_FS,
            "role": "gradient",
        })
    if step_size is not None:
        blocks.append({
            **row,
            "mathtext_lines": [rf"$\alpha = {float(step_size):.4f}$"],
            "mathtext_usetex": True,
            "mathtext_fs": CH4_HERE_BLOCK_FS,
            "role": "gradient",
        })
    return blocks


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


def ch4_formula_nll_block(**kw) -> dict:
    """Shared NLL formula block (same placement in ch4_03 end + ch4_04–06)."""
    base = dict(
        mathtext_usetex=True,
        mathtext_fs=CH4_FORMULA_MATH_FS,
        mathtext_min_fs=CH4_FORMULA_MATH_FS,
        block_fs=CH4_HERE_BLOCK_FS,
        top_pad_pt=0.0,
        text_y_inset_pt=CH4_FORMULA_NLL_DROP_PT,
        align="left",
        pt_units=True,
        weight=0.30,
        formula_slot="nll",
    )
    base.update(kw)
    return _ch4_story_mathtext_block(CH4_NLL_FORMULA_TEX, **base)


def ch4_formula_blocks_nll_story() -> list[dict]:
    return [ch4_formula_nll_block()]


def ch4_formula_grad_unit_block(**kw) -> dict:
    """Three ∂NLL formulas stacked as one movable unit (ch4_06)."""
    base = dict(
        mathtext_usetex=True,
        mathtext_fs=CH4_FORMULA_GRAD_MATH_FS,
        mathtext_min_fs=round(CH4_FORMULA_GRAD_MATH_FS * 0.82, 1),
        block_fs=CH4_HERE_BLOCK_FS,
        top_pad_pt=0.0,
        text_y_inset_pt=CH4_FORMULA_GRAD_UNIT_DROP_PT,
        text_x_frac=0.04,
        line_dy_pt=CH4_FORMULA_GRAD_LINE_DY_PT,
        align="left",
        pt_units=True,
        role="formula",
        formula_slot="grad",
        formula_grad_shift_in=CH4_FORMULA_GRAD_UNIT_SHIFT_IN,
        mathtext_lines=[
            CH4_GRAD_PARTIAL_ST_TEX,
            CH4_GRAD_PARTIAL_EL_TEX,
            CH4_GRAD_PARTIAL_B_TEX,
        ],
    )
    base.update(kw)
    return base


def ch4_formula_update_block(**kw) -> dict:
    base = dict(
        mathtext_usetex=True,
        mathtext_fs=CH4_FORMULA_GRAD_MATH_FS,
        mathtext_min_fs=round(CH4_FORMULA_GRAD_MATH_FS * 0.78, 1),
        block_fs=CH4_HERE_BLOCK_FS,
        top_pad_pt=0.0,
        text_y_inset_pt=CH4_FORMULA_GRAD_UNIT_DROP_PT,
        text_x_frac=0.02,
        line_dy_pt=CH4_FORMULA_GRAD_LINE_DY_PT,
        align="left",
        pt_units=True,
        role="formula",
        formula_slot="update",
        formula_grad_shift_in=CH4_FORMULA_GRAD_UNIT_SHIFT_IN,
        mathtext_lines=[
            CH4_GRAD_UPDATE_ST_TEX,
            CH4_GRAD_UPDATE_EL_TEX,
            CH4_GRAD_UPDATE_B_TEX,
        ],
    )
    base.update(kw)
    return base


def ch4_notation_corner_blocks() -> list[dict]:
    return [_ch4_notation_yi_xi_block(text_y_inset_pt=CH4_NOTATION_TEXT_DROP_PT)]


def ch4_formula_blocks_3d_story() -> list[dict]:
    return [
        ch4_formula_nll_block(),
        _ch4_story_mathtext_block(
            r"$p(y_i \mid x_i) = \begin{cases}"
            r"\sigma(\boldsymbol{w_{\mathrm{ST}}} x_{i,\mathrm{ST}} + \boldsymbol{w_{\mathrm{EL}}} x_{i,\mathrm{EL}} + \boldsymbol{b})"
            r" & \text{if student i passed} \\[1.0em]"
            r"1 - \sigma(\boldsymbol{w_{\mathrm{ST}}} x_{i,\mathrm{ST}} + \boldsymbol{w_{\mathrm{EL}}} x_{i,\mathrm{EL}} + \boldsymbol{b})"
            r" & \text{if student i failed} \end{cases}$",
            mathtext_fs=CH4_FORMULA_MATH_FS,
            mathtext_min_fs=CH4_FORMULA_MATH_FS,
            block_fs=CH4_HERE_BLOCK_FS,
            top_pad_pt=0.0,
            text_y_inset_pt=-40.0 + CH4_FORMULA_PROB_DROP_PT,
            text_x_frac=CH4_FORMULA_PROB_TEXT_X_FRAC,
            align="left",
            weight=0.70,
        ),
    ]


def ch4_formula_blocks_gd_story(*, bold_update_idx: int | None = None) -> list[dict]:
    updates = [CH4_GRAD_UPDATE_ST_TEX, CH4_GRAD_UPDATE_EL_TEX, CH4_GRAD_UPDATE_B_TEX]
    if bold_update_idx is not None:
        i = int(bold_update_idx)
        if 0 <= i < len(updates):
            updates = list(updates)
            updates[i] = ch4_bold_mathtext(updates[i])
    return [
        ch4_formula_nll_block(),
        ch4_formula_grad_unit_block(mathtext_line_colors=list(CH4_GD_PARTIAL_COLORS)),
        ch4_formula_update_block(mathtext_lines=updates),
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
    rails_cache_key=None,
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
    )


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
