"""Fixed-layout tutorial frame: plot slot + math rails, typography, themes, gradients."""
from __future__ import annotations

import io
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import handwrite_tutorial as hw


@dataclass(frozen=True)
class TutorialExport:
    figsize: tuple[float, float] = (15.0, 9.5)
    dpi: int = 200
    pad_inches: float = 0.10


@dataclass(frozen=True)
class TutorialLayout:
    plot_base_frac: float = 2.0 / 3.0
    plot_scale: float = 1.20
    region_pad: float = 0.010
    right_gap_frac: float = 0.001
    fig_edge_pad: float = 0.004
    section_title_gap_frac: float = 0.040
    right_section_title_gap_frac: float = 0.010
    right_section_drop_frac: float = 0.038
    right_rail_lower_frac: float = 0.022
    bottom_col_gap_frac: float = 0.036
    bottom_col_pad_frac: float = 0.012
    right_row_gap_frac: float = 0.024
    corner_row_gap_frac: float = 0.008
    right_row_pad_frac: float = 0.010
    bottom_plot_gap_frac: float = 0.001
    bottom_section_drop_frac: float = 0.014
    right_text_inset_frac: float = 0.014
    bottom_text_drop_frac: float = 0.010
    corner_title_lift_in: float = 3.0 / 25.4  # raise Notation title ~3 mm (figure inches)

    @property
    def plot_w_frac(self) -> float:
        return self.plot_base_frac * self.plot_scale

    @property
    def plot_h_frac(self) -> float:
        return self.plot_base_frac * self.plot_scale


@dataclass(frozen=True)
class TutorialTypography:
    right_block_fs: float = 28.75
    right_label_fs: float = 22.36
    bottom_block_fs: float = 28.12
    bottom_label_fs: float = 21.73
    right_section_title_fs: float = 26.14
    bottom_section_title_fs: float = 31.94
    label_gap_pt: float = 20.0
    line_dy_pt: float = 30.0
    title_line_dy_pt: float = 6.0
    top_pad_pt: float = 3.0
    bottom_pad_pt: float = 5.0
    reveal_overlap: float = 0.28
    title_reveal_boost: float = 1.4

    def handwrite_style(self, theme: TutorialTheme) -> hw.HandwriteStyle:
        return hw.HandwriteStyle(
            enabled=True,
            section_title_fs=self.bottom_section_title_fs,
            line_mode="char",
            title_mode="word",
            label_gap_pt=self.label_gap_pt,
            line_dy_pt=self.line_dy_pt,
            top_pad_pt=self.top_pad_pt,
            bottom_pad_pt=self.bottom_pad_pt,
            title_line_dy_pt=self.title_line_dy_pt,
            label_color=theme.label_color,
            text_color=theme.text_color,
            title_color=theme.bottom_title_color,
            accent_color=theme.accent_color,
            frame_edge=theme.frame_edge or "none",
            frame_lw=0.0 if not theme.frame_edge else 0.7,
            reveal_overlap=self.reveal_overlap,
            title_reveal_boost=self.title_reveal_boost,
            section_title_bold=True,
        )


@dataclass(frozen=True)
class TutorialTheme:
    name: str
    fig_bg: str
    plot_vignette: str
    right_grad: tuple[str, str]
    bottom_grad: tuple[str, str]
    right_title_color: str
    bottom_title_color: str
    label_color: str
    text_color: str
    accent_color: str
    frame_edge: str | None = None
    crossfade_frac: float = 0.055
    fade_from: str = "#ffffff"
    white_fade_frac: float = 0.10
    gradient_label_color: str | None = None
    nll_accent_color: str | None = None
    formula_accent_color: str | None = None

    def rail_fill(self) -> str:
        """Solid panel color shared by row and column rails."""
        return self.right_grad[1]

    def panel_fade_from(self) -> str:
        """Plot-facing edge of the panel gradient (lighter rail tone)."""
        return self.right_grad[0]

    def block_colors(self, block: dict, *, region: str) -> dict[str, str]:
        role = str(block.get("role", ""))
        out = {"label_color": self.label_color, "text_color": self.text_color, "accent_color": self.accent_color}
        if role == "gradient" and self.gradient_label_color:
            out["label_color"] = self.gradient_label_color
            out["accent_color"] = self.gradient_label_color
        elif role == "nll" and self.nll_accent_color:
            out["label_color"] = self.nll_accent_color
            out["accent_color"] = self.nll_accent_color
        elif role == "formula" and self.formula_accent_color:
            out["accent_color"] = self.formula_accent_color
        elif role == "weights" and region == "right":
            out["label_color"] = self.label_color
        return out


@dataclass
class TutorialScene:
    plot: Any
    math_right_blocks: list[dict] | None = None
    math_bottom_blocks: list[dict] | None = None
    math_corner_blocks: list[dict] | None = None
    right_section_title: str = "Gradient Descent Step"
    bottom_section_title: str = "Key formulas"
    corner_section_title: str = "Notation"
    right_title_color: str | None = None


DEFAULT_EXPORT = TutorialExport()
DEFAULT_LAYOUT = TutorialLayout()
DEFAULT_TYPOGRAPHY = TutorialTypography()


def _hex_rgb(hex_color: str) -> tuple[float, float, float]:
    s = str(hex_color).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return tuple(int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _lerp_rgb(c0, c1, t):
    return tuple(c0[i] * (1.0 - t) + c1[i] * t for i in range(3))


def _ease_out_quad(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 1.0 - (1.0 - t) ** 2


def _solid_rgba(w: int, h: int, color: str) -> np.ndarray:
    w, h = max(int(w), 2), max(int(h), 2)
    rgb = np.array(_hex_rgb(color), dtype=np.float32)
    arr = np.zeros((h, w, 4), dtype=np.float32)
    arr[:, :, :3] = rgb
    arr[:, :, 3] = 1.0
    return arr


def _panel_bg_rgba(
    w: int,
    h: int,
    target: str,
    *,
    panel: str,
    fade_from: str,
    fade_frac: float,
    overlap_right_x: float | None = None,
    overlap_bottom_y: float | None = None,
    rect_x0: float = 0.0,
    rect_y0: float = 0.0,
    rect_w: float = 1.0,
    rect_h: float = 1.0,
) -> np.ndarray:
    """Build panel background: white fade from left (row) or top (col); solid in overlap."""
    w, h = max(int(w), 2), max(int(h), 2)
    rgb_t = np.array(_hex_rgb(target), dtype=np.float32)
    rgb_w = np.array(_hex_rgb(fade_from), dtype=np.float32)
    arr = np.zeros((h, w, 4), dtype=np.float32)
    fade_px = max(int((w if panel == "row" else h) * fade_frac), 2)
    for j in range(h):
        fy = rect_y0 + rect_h * (1.0 - (j + 0.5) / h)
        for i in range(w):
            fx = rect_x0 + rect_w * ((i + 0.5) / w)
            in_overlap = (
                overlap_right_x is not None
                and overlap_bottom_y is not None
                and fx >= overlap_right_x
                and fy <= overlap_bottom_y
            )
            if in_overlap:
                rgb = rgb_t
            elif panel == "row":
                t = _ease_out_quad(min(i / fade_px, 1.0))
                rgb = rgb_w * (1.0 - t) + rgb_t * t
            else:
                t = _ease_out_quad(min(j / fade_px, 1.0))
                rgb = rgb_w * (1.0 - t) + rgb_t * t
            arr[j, i, :3] = rgb
            arr[j, i, 3] = 1.0
    return arr


def _cell_bg_rgba(w: int, h: int, target: str, *, panel: str, fade_from: str, fade_frac: float) -> np.ndarray:
    """Block cell: quick white fade from left (row) or top (column)."""
    w, h = max(int(w), 2), max(int(h), 2)
    rgb_t = np.array(_hex_rgb(target), dtype=np.float32)
    rgb_w = np.array(_hex_rgb(fade_from), dtype=np.float32)
    arr = np.zeros((h, w, 4), dtype=np.float32)
    fade_px = max(int((w if panel == "row" else h) * fade_frac), 2)
    if panel == "row":
        xs = np.arange(w, dtype=np.float32)
        t = np.clip(xs / fade_px, 0.0, 1.0)
        t = 1.0 - (1.0 - t) ** 2
        rgb = rgb_w * (1.0 - t[np.newaxis, :, np.newaxis]) + rgb_t * t[np.newaxis, :, np.newaxis]
        arr[:, :, :3] = rgb
    else:
        ys = np.arange(h, dtype=np.float32)
        t = np.clip(ys / fade_px, 0.0, 1.0)
        t = 1.0 - (1.0 - t) ** 2
        rgb = rgb_w * (1.0 - t[:, np.newaxis, np.newaxis]) + rgb_t * t[:, np.newaxis, np.newaxis]
        arr[:, :, :3] = rgb
    arr[:, :, 3] = 1.0
    return arr


def _fill_ax_white_fade(ax, target: str, *, panel: str, fade_from: str, fade_frac: float):
    img = _cell_bg_rgba(96, 96, target, panel=panel, fade_from=fade_from, fade_frac=fade_frac)
    ax.imshow(img, extent=[0, 1, 0, 1], aspect="auto", origin="upper", interpolation="bilinear", zorder=0)


def _draw_panel_bg(
    fig,
    rect,
    target: str,
    *,
    panel: str,
    fade_from: str,
    fade_frac: float,
    overlap_right_x: float | None = None,
    overlap_bottom_y: float | None = None,
    zorder=-5,
    panel_alpha: float = 1.0,
):
    x0, y0, w, h = rect
    if w <= 0 or h <= 0:
        return
    ax = fig.add_axes(rect, zorder=zorder)
    ax.set_axis_off()
    img = _panel_bg_rgba(
        160, 160, target, panel=panel, fade_from=fade_from, fade_frac=fade_frac,
        overlap_right_x=overlap_right_x, overlap_bottom_y=overlap_bottom_y,
        rect_x0=x0, rect_y0=y0, rect_w=w, rect_h=h,
    )
    img[..., 3] *= float(np.clip(panel_alpha, 0.0, 1.0))
    ax.imshow(img, extent=[0, 1, 0, 1], aspect="auto", origin="upper", interpolation="bilinear")


def _gradient_rgba(w: int, h: int, c0: str, c1: str, *, horizontal: bool = False) -> np.ndarray:
    w, h = max(int(w), 2), max(int(h), 2)
    rgb0, rgb1 = _hex_rgb(c0), _hex_rgb(c1)
    arr = np.zeros((h, w, 4), dtype=np.float32)
    for i in range(h if not horizontal else w):
        t = i / max((h if not horizontal else w) - 1, 1)
        rgb = _lerp_rgb(rgb0, rgb1, t)
        if horizontal:
            arr[:, i, :3] = rgb
            arr[:, i, 3] = 1.0
        else:
            arr[i, :, :3] = rgb
            arr[i, :, 3] = 1.0
    return arr


def _apply_edge_fade(img: np.ndarray, fade: float) -> np.ndarray:
    if fade <= 0:
        return img
    h, w = img.shape[:2]
    fx = max(int(w * fade), 1)
    fy = max(int(h * fade), 1)
    alpha = img[:, :, 3].copy()
    for i in range(fx):
        t = (i + 1) / fx
        alpha[:, i] *= t
        alpha[:, w - 1 - i] *= t
    for j in range(fy):
        t = (j + 1) / fy
        alpha[j, :] *= t
        alpha[h - 1 - j, :] *= t
    img[:, :, 3] = alpha
    return img


def _fill_ax_gradient(ax, c0: str, c1: str, *, horizontal: bool = False, fade: float = 0.10):
    img = _gradient_rgba(96, 96, c0, c1, horizontal=horizontal)
    img = _apply_edge_fade(img, fade)
    ax.imshow(img, extent=[0, 1, 0, 1], aspect="auto", origin="upper", interpolation="bilinear", zorder=0)


def _draw_gradient_rect(fig, rect, c0: str, c1: str, *, horizontal: bool = False, fade: float = 0.10, zorder=-5):
    x0, y0, w, h = rect
    if w <= 0 or h <= 0:
        return
    ax = fig.add_axes(rect, zorder=zorder)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    img = _gradient_rgba(128, 128, c0, c1, horizontal=horizontal)
    img = _apply_edge_fade(img, fade)
    ax.imshow(img, extent=[0, 1, 0, 1], aspect="auto", origin="upper", interpolation="bilinear")


def _draw_crossfade(fig, rect, c_left: str, c_right: str, *, zorder=-4):
    x0, y0, w, h = rect
    if w <= 0 or h <= 0:
        return
    ax = fig.add_axes(rect, zorder=zorder)
    ax.set_axis_off()
    rgb_l, rgb_r = _hex_rgb(c_left), _hex_rgb(c_right)
    w_px = 64
    strip = np.zeros((2, w_px, 4), dtype=np.float32)
    for i in range(w_px):
        t = i / max(w_px - 1, 1)
        rgb = _lerp_rgb(rgb_l, rgb_r, t)
        strip[0, i, :3] = rgb
        strip[1, i, :3] = rgb
        strip[0, i, 3] = strip[1, i, 3] = 1.0
    ax.imshow(strip, extent=[0, 1, 0, 1], aspect="auto", origin="upper", interpolation="bicubic")


class TutorialComposer:
    def __init__(
        self,
        *,
        export: TutorialExport = DEFAULT_EXPORT,
        layout: TutorialLayout = DEFAULT_LAYOUT,
        typography: TutorialTypography = DEFAULT_TYPOGRAPHY,
        theme: TutorialTheme | None = None,
    ):
        self.export = export
        self.layout = layout
        self.typography = typography
        self.theme = theme or TUTORIAL_THEMES["classic_light"]
        self._rect_cache: dict | None = None
        self._rails_overlay_cache: dict[str, Image.Image] = {}
        self._bottom_block_rects_cache: dict[str, list] = {}
        self._corner_block_rects_cache: dict[str, list] = {}
        self._right_block_rects_cache: dict[str, list] = {}

    def clear_rails_cache(self) -> None:
        """Drop cached static rail layers and block layout rects."""
        self._rails_overlay_cache.clear()
        self._bottom_block_rects_cache.clear()
        self._corner_block_rects_cache.clear()
        self._right_block_rects_cache.clear()

    def handwrite_style(self) -> hw.HandwriteStyle:
        return self.typography.handwrite_style(self.theme)

    def layout_rects(self, fig=None) -> dict:
        if self._rect_cache is not None and fig is None:
            return self._rect_cache
        L = self.layout
        pad = float(L.region_pad)
        gap_r = float(L.right_gap_frac)
        gap_b = float(L.bottom_plot_gap_frac)
        pw, ph = L.plot_w_frac, L.plot_h_frac
        plot_x0 = 0.0
        plot_w = pw
        plot_h = ph
        plot_y0 = 1.0 - plot_h
        plot_top = plot_y0 + plot_h
        plot = (plot_x0, plot_y0, plot_w, plot_h)
        plot_right = plot_x0 + plot_w
        plot_bottom = plot_y0
        right_panel_x0 = plot_right + gap_r
        right_panel_w = max(1.0 - right_panel_x0, 0.04)
        bottom_panel_h = max(plot_bottom - gap_b, 0.04)
        right_x0 = right_panel_x0
        right_w = right_panel_w
        bottom_y0 = 0.0
        bottom_h_total = bottom_panel_h
        bottom_top = bottom_y0 + bottom_h_total
        bottom_title_y = bottom_top - float(L.bottom_section_drop_frac) - float(L.bottom_text_drop_frac)
        style = self.handwrite_style()
        title_band_r = float(L.section_title_gap_frac) + 0.018
        title_band_b = float(L.section_title_gap_frac) + 0.018
        right_title_band_scale = 0.38
        if fig is not None:
            title_band_r = hw.section_title_band_frac(
                fig, "Gradient Descent Step", right_w,
                style=style, gap_frac=L.right_section_title_gap_frac * 0.22,
                title_fs=self.typography.right_section_title_fs,
            )
            title_band_b = hw.section_title_band_frac(
                fig, "Key formulas", 1.0 - pad,
                style=style, gap_frac=L.section_title_gap_frac,
                title_fs=self.typography.bottom_section_title_fs,
            )
        T = self.theme
        drop_r = float(L.right_section_drop_frac) + float(L.right_rail_lower_frac)
        rail_shift = float(L.right_rail_lower_frac)
        right_title_y = plot_top - drop_r
        right_content_h = max(plot_h - drop_r - title_band_r * right_title_band_scale, 0.05)
        right_content_y0 = plot_y0 - rail_shift
        right_content_h = max(right_content_h - rail_shift * 0.5, 0.04)
        bottom_content_y0 = bottom_y0 + float(L.bottom_section_drop_frac) * 0.06 + float(L.bottom_text_drop_frac) * 0.18
        bottom_title_overlap = title_band_b * 0.12
        bottom_content_y0 += bottom_title_overlap
        bottom_content_h = max(
            bottom_h_total - title_band_b * 0.58 - float(L.bottom_section_drop_frac) * 0.06
            - float(L.bottom_text_drop_frac) * 0.18 + bottom_title_overlap,
            0.05,
        )
        right_inset = float(L.right_text_inset_frac)
        corner_x0 = right_panel_x0
        corner_w = right_panel_w
        corner_h = bottom_h_total
        corner_title_y = bottom_title_y + float(L.corner_title_lift_in) / float(self.export.figsize[1])
        corner_gap_below_title = title_band_b * 0.20
        corner_content_top = bottom_title_y - corner_gap_below_title
        corner_content_y0 = bottom_content_y0
        corner_content_h = max(corner_content_top - corner_content_y0, 0.04)
        bottom_formulas_w = max(right_panel_x0 - pad - float(L.bottom_col_gap_frac), 0.12)
        rects = {
            "plot": plot,
            "math_right_title": (right_x0 + right_inset, right_title_y),
            "math_right_title_width": right_w - right_inset,
            "math_right_panel": (right_panel_x0, 0.0, right_panel_w, 1.0),
            "math_right_content": (right_x0, right_content_y0, right_w, right_content_h),
            "math_bottom_title": (pad, bottom_title_y),
            "math_bottom_title_width": bottom_formulas_w,
            "math_bottom_panel": (0.0, 0.0, 1.0, bottom_panel_h),
            "math_bottom_content": (pad, bottom_content_y0, bottom_formulas_w, bottom_content_h),
            "math_corner_title": (corner_x0 + right_inset, corner_title_y),
            "math_corner_title_width": corner_w - right_inset,
            "math_corner_panel": (corner_x0, 0.0, corner_w, corner_h),
            "math_corner_content": (corner_x0 + right_inset * 0.5, corner_content_y0, corner_w - right_inset, corner_content_h),
            "plot_right_edge": (plot_right, plot_y0, T.crossfade_frac, plot_h),
        }
        if fig is None:
            self._rect_cache = rects
        return rects

    def _draw_backgrounds(self, fig, rects, *, panel_u: float = 1.0):
        T = self.theme
        rail_color = T.rail_fill()
        fade_from = T.panel_fade_from()
        fig.patch.set_facecolor(rail_color)
        if float(panel_u) <= 1e-4:
            return
        rx, _, _, _ = rects["math_right_panel"]
        _, _, _, bh = rects["math_bottom_panel"]
        overlap_right_x = float(rx)
        overlap_bottom_y = float(bh)
        _draw_panel_bg(
            fig, rects["math_right_panel"], rail_color,
            panel="row", fade_from=fade_from, fade_frac=T.white_fade_frac,
            overlap_right_x=overlap_right_x, overlap_bottom_y=overlap_bottom_y,
            panel_alpha=float(panel_u),
        )
        _draw_panel_bg(
            fig, rects["math_bottom_panel"], rail_color,
            panel="column", fade_from=fade_from, fade_frac=T.white_fade_frac,
            overlap_right_x=overlap_right_x, overlap_bottom_y=overlap_bottom_y,
            panel_alpha=float(panel_u),
        )

    def _column_block_rects(self, fig, blocks, panel_rect, *, gap_frac: float, pad_frac: float, default_fs: float):
        """Lay out blocks left-to-right in one row (bottom rail or corner panel)."""
        blocks = list(blocks)
        if not blocks:
            return []
        x0, y0, total_w, h = (float(v) for v in panel_rect)
        gap = float(gap_frac)
        pad_each = float(pad_frac)
        fs = float(default_fs)
        style = self.handwrite_style()

        def _text_w(block):
            bfs = float(block.get("block_fs", fs))
            renderer = fig.canvas.get_renderer()
            text = hw.block_display_text(block, style=style)
            if text:
                return hw.mixed_line_width_px(
                    renderer,
                    hw.parse_handwrite_runs(text),
                    bfs, style=style, bold=False, fp_hand=hw.hand_font(bfs),
                ) / float(fig.bbox.width)
            mlines = hw.block_mathtext_lines(block)
            if not mlines:
                return pad_each
            m_fs = float(block.get("mathtext_fs", fs))
            return max(
                hw.mathtext_line_width_px(
                    renderer, line, m_fs, usetex=hw.block_mathtext_usetex(block),
                ) / float(fig.bbox.width)
                for line in mlines
            )

        fig.canvas.draw()
        raw = []
        for block in blocks:
            w = float(block.get("weight", 0.0))
            if w <= 0.0:
                w = _text_w(block)
            raw.append(max(w + 2.0 * pad_each, pad_each))
        gaps_total = gap * max(len(raw) - 1, 0)
        avail = max(total_w - gaps_total, 1e-6)
        scale = avail / max(sum(raw), 1e-6)
        rects, x = [], x0
        for w in raw:
            cw = w * scale
            rects.append((x, y0, cw, h))
            x += cw + gap
        return rects

    def _bottom_gd_column_rects(self, blocks, bottom_rect):
        """NLL column + stacked ∂ column + update rule (ch4_06 formulas)."""
        x0, y0, total_w, h = (float(v) for v in bottom_rect)
        gap = float(self.layout.bottom_col_gap_frac)
        shift_in = 0.0
        for b in blocks[1:3]:
            shift_in = max(shift_in, float(b.get("formula_grad_shift_in", 0.0)))
        shift_frac = shift_in / float(self.export.figsize[0])
        nll_w = max(total_w * 0.20, 0.12)
        upd_w = max(total_w * 0.30, 0.18)
        grad_w = max(total_w - nll_w - upd_w - 2.0 * gap - shift_frac, 0.26)
        grad_x = x0 + nll_w + gap + shift_frac
        upd_x = grad_x + grad_w + gap
        return [
            (x0, y0, nll_w, h),
            (grad_x, y0, grad_w, h),
            (upd_x, y0, max(total_w - (upd_x - x0), upd_w), h),
        ]

    def _bottom_block_rects(self, fig, blocks, bottom_rect):
        blocks = list(blocks)
        if (
            len(blocks) == 3
            and blocks[0].get("formula_slot") == "nll"
            and blocks[1].get("formula_slot") == "grad"
            and blocks[2].get("formula_slot") == "update"
        ):
            return self._bottom_gd_column_rects(blocks, bottom_rect)
        return self._column_block_rects(
            fig, blocks, bottom_rect,
            gap_frac=self.layout.bottom_col_gap_frac,
            pad_frac=self.layout.bottom_col_pad_frac,
            default_fs=self.typography.bottom_block_fs,
        )

    def _corner_block_rects(self, fig, blocks, corner_rect):
        return self._column_block_rects(
            fig, blocks, corner_rect,
            gap_frac=self.layout.bottom_col_gap_frac,
            pad_frac=self.layout.bottom_col_pad_frac * 0.6,
            default_fs=self.typography.right_block_fs,
        )

    def _right_block_rects(self, fig, blocks, right_rect, *, row_gap_frac=None):
        blocks = list(blocks)
        if not blocks:
            return []
        x0, y0, w, total_h = (float(v) for v in right_rect)
        gap = float(row_gap_frac if row_gap_frac is not None else self.layout.right_row_gap_frac)
        pad_each = float(self.layout.right_row_pad_frac)
        fs = float(self.typography.right_block_fs)
        lbl = float(self.typography.right_label_fs)
        style = self.handwrite_style()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        raw = []
        for block in blocks:
            h = float(block.get("weight", 0.0))
            if h <= 0.0:
                st = style
                total_px = float(block.get("top_pad_pt", st.top_pad_pt)) + float(st.bottom_pad_pt)
                if block.get("label"):
                    total_px += hw.text_line_height_px(renderer, str(block["label"]), lbl, style=st)
                    total_px += float(st.label_gap_pt)
                for line in hw.block_display_lines(block, style=st):
                    runs = hw.parse_handwrite_runs(line)
                    fp = hw.hand_font(fs)
                    total_px += hw.mixed_line_height_px(renderer, runs, fs, style=st, bold=False, fp_hand=fp)
                    total_px += float(st.line_dy_pt)
                m_fs_raw = block.get("mathtext_fs", fs)
                m_fs_list = list(m_fs_raw) if isinstance(m_fs_raw, (list, tuple)) else None
                for j, mline in enumerate(hw.block_mathtext_lines(block)):
                    m_fs = float(m_fs_list[j] if m_fs_list and j < len(m_fs_list) else m_fs_raw)
                    total_px += hw.mathtext_line_height_px(
                        renderer, mline, m_fs, usetex=hw.block_mathtext_usetex(block),
                    )
                    total_px += float(block.get("line_dy_pt", st.line_dy_pt))
                h = total_px / float(fig.bbox.height)
            raw.append(max(h + 2.0 * pad_each, pad_each))
        gaps_total = gap * max(len(raw) - 1, 0)
        need_h = sum(raw) + gaps_total
        scale = 1.0 if need_h <= total_h else min(1.0, max(total_h, 1e-6) / max(need_h, 1e-6))
        rects, y = [], y0 + total_h
        for i, (block, h) in enumerate(zip(blocks, raw)):
            rh = h * scale
            y -= rh
            rects.append((x0, y, w, rh))
            if i < len(blocks) - 1:
                y -= gap
                pre_gap_pt = float(blocks[i + 1].get("pre_gap_pt", 0.0))
                if pre_gap_pt:
                    y -= hw.block_pt_to_px(fig, blocks[i + 1], pre_gap_pt) / float(fig.bbox.height)
        return rects

    def _write_progress_groups(
        self,
        right_blocks,
        bottom_blocks,
        progress,
        *,
        corner_blocks=None,
        right_progress=None,
        bottom_progress=None,
        corner_progress=None,
    ):
        style = self.handwrite_style()
        groups, keys = [], []
        if right_blocks:
            groups.append(right_blocks)
            keys.append("right")
        if bottom_blocks:
            groups.append(bottom_blocks)
            keys.append("bottom")
        if corner_blocks:
            groups.append(corner_blocks)
            keys.append("corner")
        if not groups:
            return {}
        if right_progress is not None or bottom_progress is not None or corner_progress is not None:
            per = []
            for key, blocks in zip(keys, groups):
                p = float(progress)
                if key == "right" and right_progress is not None:
                    p = float(right_progress)
                if key == "bottom" and bottom_progress is not None:
                    p = float(bottom_progress)
                if key == "corner" and corner_progress is not None:
                    p = float(corner_progress)
                per.append(hw.block_write_progress(blocks, p, style=style))
            return dict(zip(keys, per))
        if len(groups) == 1:
            return {keys[0]: hw.block_write_progress(groups[0], progress, style=style)}
        per = hw.stagger_groups_progress(groups, progress, style=style)
        return dict(zip(keys, per))

    @staticmethod
    def _lerp_plot_rect(start, end, u: float):
        su = float(np.clip(float(u), 0.0, 1.0))
        su = 1.0 - (1.0 - su) ** 2
        return tuple(float(a + (b - a) * su) for a, b in zip(start, end))

    def _title_u(self, panel_u: float, title_write_progress: float | None) -> float:
        title_u = float(panel_u) if float(panel_u) < 1.0 - 1e-9 else 1.0
        if title_write_progress is not None:
            title_u = float(np.clip(float(title_write_progress), 0.0, 1.0))
        return title_u

    def _render_static_rails(
        self,
        scene: TutorialScene,
        *,
        panel_u: float,
        title_u: float,
        cache_key: str,
    ) -> Image.Image:
        """Backgrounds, section titles, bottom formulas, corner notation (no plot / right values)."""
        if cache_key in self._rails_overlay_cache:
            return self._rails_overlay_cache[cache_key]

        T = self.theme
        canvas_bg = T.rail_fill()
        fig = plt.figure(figsize=self.export.figsize, facecolor=canvas_bg)
        rects = self.layout_rects(fig)
        self._draw_backgrounds(fig, rects, panel_u=float(panel_u))
        style = self.handwrite_style()
        ty = self.typography

        if scene.math_right_blocks and float(panel_u) > 1e-4:
            tw = rects["math_right_title_width"]
            rt_color = scene.right_title_color if scene.right_title_color else T.right_title_color
            hw.draw_section_title(
                fig, rects["math_right_title"], scene.right_section_title,
                style=style, ha="left", va="top", max_width_frac=tw,
                write_progress=min(1.0, title_u * ty.title_reveal_boost),
                pad_frac=self.layout.region_pad * 0.55,
                title_fs=ty.right_section_title_fs,
                title_color=rt_color,
            )

        if scene.math_corner_blocks and float(panel_u) > 1e-4:
            ctw = rects["math_corner_title_width"]
            hw.draw_section_title(
                fig, rects["math_corner_title"], scene.corner_section_title,
                style=style, ha="left", va="top", max_width_frac=ctw,
                write_progress=min(1.0, title_u),
                pad_frac=self.layout.region_pad * 0.18,
                title_fs=ty.right_section_title_fs,
                title_color=T.bottom_title_color,
            )
            corner_rects = self._corner_block_rects_cache.get(cache_key)
            if corner_rects is None:
                corner_rects = self._right_block_rects(
                    fig, scene.math_corner_blocks, rects["math_corner_content"],
                    row_gap_frac=self.layout.corner_row_gap_frac,
                )
                self._corner_block_rects_cache[cache_key] = corner_rects
            for bi, (block, rect) in enumerate(zip(scene.math_corner_blocks, corner_rects)):
                ax = fig.add_axes(rect, zorder=2, facecolor="none")
                ax.set_clip_on(False)
                colors = T.block_colors(block, region="right")
                hw.draw_block_cell(
                    ax, block, style=style,
                    block_fs=float(block.get("block_fs", ty.right_block_fs)),
                    label_fs=float(block.get("label_fs", ty.right_label_fs)),
                    align=str(block.get("align", "left")),
                    line_progress={}, show_frame=bool(T.frame_edge),
                    text_x_frac=0.04,
                    text_y_inset_pt=float(block.get("text_y_inset_pt", 0.0)),
                    **colors,
                )

        if scene.math_bottom_blocks and float(panel_u) > 1e-4:
            tw = rects["math_bottom_title_width"]
            hw.draw_section_title(
                fig, rects["math_bottom_title"], scene.bottom_section_title,
                style=style, ha="left", va="top", max_width_frac=tw,
                write_progress=min(1.0, title_u),
                pad_frac=self.layout.region_pad,
                title_fs=ty.bottom_section_title_fs, title_color=T.bottom_title_color,
            )
            bottom_rects = self._bottom_block_rects_cache.get(cache_key)
            if bottom_rects is None:
                bottom_rects = self._bottom_block_rects(
                    fig, scene.math_bottom_blocks, rects["math_bottom_content"],
                )
                self._bottom_block_rects_cache[cache_key] = bottom_rects
            for bi, (block, rect) in enumerate(zip(scene.math_bottom_blocks, bottom_rects)):
                ax = fig.add_axes(rect, zorder=2, facecolor="none")
                if hw.block_mathtext_usetex(block):
                    ax.set_clip_on(False)
                colors = T.block_colors(block, region="bottom")
                hw.draw_block_cell(
                    ax, block, style=style,
                    block_fs=float(block.get("block_fs", ty.bottom_block_fs)),
                    label_fs=float(block.get("label_fs", ty.bottom_label_fs)),
                    align=str(block.get("align", "center")),
                    line_progress={}, show_frame=bool(T.frame_edge),
                    text_x_frac=float(block.get("text_x_frac", 0.5 if block.get("align", "center") == "center" else 0.07)),
                    text_y_inset_pt=float(block.get("text_y_inset_pt", 2.0)),
                    **colors,
                )

        overlay = self.fig_to_image(fig)
        self._rails_overlay_cache[cache_key] = overlay
        return overlay

    def _render_plot_and_right_overlay(
        self,
        scene: TutorialScene,
        *,
        plot_rect,
        plot_alpha: float,
        prog: dict,
        title_u: float,
        panel_u: float,
        right_rects_key: str | None,
    ) -> Image.Image:
        """Plot slot + We-are-here values on a transparent layer for compositing."""
        T = self.theme
        canvas_bg = T.rail_fill()
        fig = plt.figure(figsize=self.export.figsize, facecolor="none")
        fig.patch.set_alpha(0.0)
        style = self.handwrite_style()
        ty = self.typography

        if scene.plot is not None:
            ax_plot = fig.add_axes(plot_rect, zorder=1)
            pa = float(np.clip(float(plot_alpha), 0.0, 1.0))
            ax_plot.imshow(scene.plot, aspect="auto", interpolation="lanczos", alpha=pa)
            ax_plot.set_axis_off()
            ax_plot.set_facecolor("none")
            if T.frame_edge:
                for spine in ax_plot.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(T.frame_edge)
                    spine.set_linewidth(1.0)

        if scene.math_right_blocks and float(panel_u) > 1e-4:
            rects = self.layout_rects(fig)
            block_rects = None
            if right_rects_key:
                block_rects = self._right_block_rects_cache.get(right_rects_key)
            if block_rects is None:
                block_rects = self._right_block_rects(
                    fig, scene.math_right_blocks, rects["math_right_content"],
                )
                if right_rects_key:
                    self._right_block_rects_cache[right_rects_key] = block_rects
            per = prog.get("right", {})
            for bi, (block, rect) in enumerate(zip(scene.math_right_blocks, block_rects)):
                ax = fig.add_axes(rect, zorder=2, facecolor="none")
                ax.set_clip_on(False)
                colors = T.block_colors(block, region="right")
                hw.draw_block_cell(
                    ax, block, style=style,
                    block_fs=float(block.get("block_fs", ty.right_block_fs)),
                    label_fs=float(block.get("label_fs", ty.right_label_fs)),
                    align=str(block.get("align", "left")),
                    line_progress=per.get(bi, {}), show_frame=bool(T.frame_edge),
                    text_x_frac=0.07 + self.layout.right_text_inset_frac,
                    text_y_inset_pt=float(block.get("text_y_inset_pt", -5.0)),
                    **colors,
                )

        return self.fig_to_image(fig, transparent=True)

    def _compose_frame_cached_rails(
        self,
        scene: TutorialScene,
        cache_key: str,
        *,
        write_progress: float,
        panel_u: float,
        right_write_progress: float | None,
        bottom_write_progress: float | None,
        plot_alpha: float,
        title_write_progress: float | None,
    ) -> Image.Image:
        title_u = self._title_u(panel_u, title_write_progress)
        static = self._render_static_rails(
            scene, panel_u=panel_u, title_u=title_u, cache_key=cache_key,
        )
        block_write = float(write_progress) if float(panel_u) >= 1.0 - 1e-9 else 0.0
        prog = self._write_progress_groups(
            scene.math_right_blocks,
            None,
            block_write,
            corner_blocks=None,
            right_progress=right_write_progress,
            bottom_progress=bottom_write_progress,
        )
        dynamic = self._render_plot_and_right_overlay(
            scene,
            plot_rect=self.layout_rects()["plot"],
            plot_alpha=plot_alpha,
            prog=prog,
            title_u=title_u,
            panel_u=panel_u,
            right_rects_key=f"{cache_key}:right",
        )
        base = static.convert("RGBA")
        return Image.alpha_composite(base, dynamic.convert("RGBA")).convert("RGB")

    def compose_frame(
        self,
        scene: TutorialScene,
        *,
        write_progress: float = 1.0,
        layout_u: float = 1.0,
        panel_u: float = 1.0,
        plot_start_rect: tuple[float, float, float, float] | None = None,
        right_write_progress: float | None = None,
        bottom_write_progress: float | None = None,
        progress_override: dict | None = None,
        plot_alpha: float = 1.0,
        title_write_progress: float | None = None,
        rails_cache_key: str | None = None,
    ) -> Image.Image:
        if (
            rails_cache_key
            and scene.plot is not None
            and float(layout_u) >= 1.0 - 1e-9
            and plot_start_rect is None
            and progress_override is None
            and float(panel_u) >= 1.0 - 1e-9
            and float(write_progress) >= 1.0 - 1e-9
            and right_write_progress is None
            and bottom_write_progress is None
        ):
            return self._compose_frame_cached_rails(
                scene,
                rails_cache_key,
                write_progress=write_progress,
                panel_u=panel_u,
                right_write_progress=right_write_progress,
                bottom_write_progress=bottom_write_progress,
                plot_alpha=plot_alpha,
                title_write_progress=title_write_progress,
            )

        T = self.theme
        canvas_bg = T.rail_fill()
        fig = plt.figure(figsize=self.export.figsize, facecolor=canvas_bg)
        rects = self.layout_rects(fig)
        self._draw_backgrounds(fig, rects, panel_u=float(panel_u))
        style = self.handwrite_style()
        ty = self.typography

        end_plot = rects["plot"]
        if plot_start_rect is not None and float(layout_u) < 1.0 - 1e-9:
            plot_rect = self._lerp_plot_rect(plot_start_rect, end_plot, layout_u)
        else:
            plot_rect = end_plot

        ax_plot = fig.add_axes(plot_rect, zorder=1)
        pa = float(np.clip(float(plot_alpha), 0.0, 1.0))
        ax_plot.imshow(scene.plot, aspect="auto", interpolation="lanczos", alpha=pa)
        ax_plot.set_axis_off()
        ax_plot.set_facecolor(canvas_bg)
        if T.frame_edge:
            for spine in ax_plot.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(T.frame_edge)
                spine.set_linewidth(1.0)

        block_write = float(write_progress) if float(panel_u) >= 1.0 - 1e-9 else 0.0
        prog = progress_override if progress_override is not None else self._write_progress_groups(
            scene.math_right_blocks,
            scene.math_bottom_blocks,
            block_write,
            corner_blocks=scene.math_corner_blocks,
            right_progress=right_write_progress,
            bottom_progress=bottom_write_progress,
        )

        title_u = self._title_u(panel_u, title_write_progress)

        if scene.math_right_blocks and float(panel_u) > 1e-4:
            tw = rects["math_right_title_width"]
            rt_color = scene.right_title_color if scene.right_title_color else T.right_title_color
            hw.draw_section_title(
                fig, rects["math_right_title"], scene.right_section_title,
                style=style, ha="left", va="top", max_width_frac=tw,
                write_progress=min(1.0, title_u * ty.title_reveal_boost),
                pad_frac=self.layout.region_pad * 0.55,
                title_fs=ty.right_section_title_fs,
                title_color=rt_color,
            )
            per = prog.get("right", {})
            for bi, (block, rect) in enumerate(
                zip(scene.math_right_blocks, self._right_block_rects(fig, scene.math_right_blocks, rects["math_right_content"]))
            ):
                ax = fig.add_axes(rect, zorder=2, facecolor="none")
                ax.set_clip_on(False)
                colors = T.block_colors(block, region="right")
                hw.draw_block_cell(
                    ax, block, style=style,
                    block_fs=float(block.get("block_fs", ty.right_block_fs)),
                    label_fs=float(block.get("label_fs", ty.right_label_fs)),
                    align=str(block.get("align", "left")),
                    line_progress=per.get(bi, {}), show_frame=bool(T.frame_edge),
                    text_x_frac=0.07 + self.layout.right_text_inset_frac,
                    text_y_inset_pt=float(block.get("text_y_inset_pt", -5.0)),
                    **colors,
                )

        if scene.math_corner_blocks and float(panel_u) > 1e-4:
            ctw = rects["math_corner_title_width"]
            hw.draw_section_title(
                fig, rects["math_corner_title"], scene.corner_section_title,
                style=style, ha="left", va="top", max_width_frac=ctw,
                write_progress=min(1.0, title_u),
                pad_frac=self.layout.region_pad * 0.18,
                title_fs=ty.right_section_title_fs,
                title_color=T.bottom_title_color,
            )
            per = prog.get("corner", {})
            for bi, (block, rect) in enumerate(
                zip(
                    scene.math_corner_blocks,
                    self._right_block_rects(
                        fig, scene.math_corner_blocks, rects["math_corner_content"],
                        row_gap_frac=self.layout.corner_row_gap_frac,
                    ),
                )
            ):
                ax = fig.add_axes(rect, zorder=2, facecolor="none")
                ax.set_clip_on(False)
                colors = T.block_colors(block, region="right")
                hw.draw_block_cell(
                    ax, block, style=style,
                    block_fs=float(block.get("block_fs", ty.right_block_fs)),
                    label_fs=float(block.get("label_fs", ty.right_label_fs)),
                    align=str(block.get("align", "left")),
                    line_progress=per.get(bi, {}), show_frame=bool(T.frame_edge),
                    text_x_frac=0.04,
                    text_y_inset_pt=float(block.get("text_y_inset_pt", 0.0)),
                    **colors,
                )

        if scene.math_bottom_blocks and float(panel_u) > 1e-4:
            tw = rects["math_bottom_title_width"]
            hw.draw_section_title(
                fig, rects["math_bottom_title"], scene.bottom_section_title,
                style=style, ha="left", va="top", max_width_frac=tw,
                write_progress=min(1.0, title_u),
                pad_frac=self.layout.region_pad,
                title_fs=ty.bottom_section_title_fs, title_color=T.bottom_title_color,
            )
            per = prog.get("bottom", {})
            for bi, (block, rect) in enumerate(
                zip(scene.math_bottom_blocks, self._bottom_block_rects(fig, scene.math_bottom_blocks, rects["math_bottom_content"]))
            ):
                ax = fig.add_axes(rect, zorder=2, facecolor="none")
                if hw.block_mathtext_usetex(block):
                    ax.set_clip_on(False)
                colors = T.block_colors(block, region="bottom")
                hw.draw_block_cell(
                    ax, block, style=style,
                    block_fs=float(block.get("block_fs", ty.bottom_block_fs)),
                    label_fs=float(block.get("label_fs", ty.bottom_label_fs)),
                    align=str(block.get("align", "center")),
                    line_progress=per.get(bi, {}), show_frame=bool(T.frame_edge),
                    text_x_frac=float(block.get("text_x_frac", 0.5 if block.get("align", "center") == "center" else 0.07)),
                    text_y_inset_pt=float(block.get("text_y_inset_pt", 2.0)),
                    **colors,
                )

        return self.fig_to_image(fig)

    def fig_to_image(self, fig, *, transparent: bool = False) -> Image.Image:
        buf = io.BytesIO()
        save_kw = dict(format="png", dpi=self.export.dpi, edgecolor="none")
        if transparent:
            save_kw["facecolor"] = "none"
            save_kw["transparent"] = True
        else:
            save_kw["facecolor"] = self.theme.rail_fill()
        fig.savefig(buf, **save_kw)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGBA")
        target = (int(self.export.figsize[0] * self.export.dpi), int(self.export.figsize[1] * self.export.dpi))
        if img.size != target:
            img = img.resize(target, Image.Resampling.LANCZOS)
        return img

    def render_scene(
        self,
        scene: TutorialScene,
        write_progress: float = 1.0,
        **compose_kw,
    ) -> Image.Image:
        return self.compose_frame(scene, write_progress=write_progress, **compose_kw)

    def export_mp4(
        self,
        scene: TutorialScene,
        filename: str,
        *,
        save_mp4: Callable,
        output_dir: Path,
        n_frames: int = 32,
        ms_per_frame: int = 100,
    ) -> Path:
        return hw.export_handwrite_mp4(
            lambda t: self.render_scene(scene, write_progress=t),
            filename,
            save_mp4=save_mp4,
            output_dir=output_dir,
            n_frames=n_frames,
            ms_per_frame=ms_per_frame,
        )


TUTORIAL_THEMES: dict[str, TutorialTheme] = {
    "classic_light": TutorialTheme(
        name="classic_light",
        fig_bg="#f7f8fb",
        plot_vignette="#ffffff",
        right_grad=("#eef0f6", "#e6e9f0"),
        bottom_grad=("#eef0f6", "#e6e9f0"),
        right_title_color="#12121c",
        bottom_title_color="#12121c",
        label_color="#5a5a72",
        text_color="#1a1a28",
        accent_color="#2563eb",
        frame_edge=None,
        crossfade_frac=0.06,
    ),
    "dark_rails": TutorialTheme(
        name="dark_rails",
        fig_bg="#14141c",
        plot_vignette="#1c1c28",
        right_grad=("#2a2a3a", "#1e2430"),
        bottom_grad=("#2a2a3a", "#1e2430"),
        right_title_color="#f0f4ff",
        bottom_title_color="#f0f4ff",
        label_color="#94a3b8",
        text_color="#f1f5f9",
        accent_color="#38bdf8",
        gradient_label_color="#67e8f9",
        nll_accent_color="#fbbf24",
        formula_accent_color="#a5b4fc",
        frame_edge=None,
        crossfade_frac=0.07,
    ),
    "midnight_gold": TutorialTheme(
        name="midnight_gold",
        fig_bg="#0c1018",
        plot_vignette="#141c28",
        right_grad=("#1a2234", "#141820"),
        bottom_grad=("#1a2234", "#141820"),
        right_title_color="#e2e8f0",
        bottom_title_color="#e2e8f0",
        label_color="#8899aa",
        text_color="#dce4ee",
        accent_color="#fbbf24",
        gradient_label_color="#4ade80",
        nll_accent_color="#fbbf24",
        formula_accent_color="#60a5fa",
        frame_edge=None,
        crossfade_frac=0.065,
    ),
    "forest_gd": TutorialTheme(
        name="forest_gd",
        fig_bg="#f2f6f2",
        plot_vignette="#fafcfa",
        right_grad=("#dcebe0", "#d0e4d8"),
        bottom_grad=("#dcebe0", "#d0e4d8"),
        right_title_color="#1a2820",
        bottom_title_color="#1a2820",
        label_color="#4a6050",
        text_color="#1a2820",
        accent_color="#059669",
        gradient_label_color="#059669",
        nll_accent_color="#d97706",
        formula_accent_color="#0284c7",
        frame_edge=None,
        crossfade_frac=0.06,
    ),
    "sunset_formula": TutorialTheme(
        name="sunset_formula",
        fig_bg="#fff9f5",
        plot_vignette="#fffdfb",
        right_grad=("#e8ddd8", "#d4dce8"),
        bottom_grad=("#e8ddd8", "#d4dce8"),
        right_title_color="#3d2010",
        bottom_title_color="#102040",
        label_color="#8a5040",
        text_color="#281810",
        accent_color="#ea580c",
        gradient_label_color="#c2410c",
        nll_accent_color="#9333ea",
        formula_accent_color="#2563eb",
        frame_edge=None,
        crossfade_frac=0.07,
    ),
}


def make_composer(theme_name: str = "classic_light") -> TutorialComposer:
    theme = TUTORIAL_THEMES[theme_name]
    return TutorialComposer(theme=theme)


def list_themes() -> list[str]:
    return list(TUTORIAL_THEMES.keys())
