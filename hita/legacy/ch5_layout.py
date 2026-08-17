"""Chapter 6 — grids, badge, Ch5-style frames (no formula rails)."""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from ch4_layout import CH4_FIGSIZE, OUTPUT_DIR, ch4_nll_heatmap_cmap
from ch6_layout import CH6_EXPORT_DPI, CH6_FIGSIZE
from ch5_core import (
    CH5_BELIEF_SURFACE_ALPHA,
    CH5_DENSITY_Z_HI,
    ch5_clip_belief_height,
    ch5_global_density_limits,
    ch5_prior_w12_z_lim,
)
from ch5_datasets import CH5_DATASET_KEYS

_CH5_GRID_SLOTS = {"D1": (0, 0), "D2": (0, 1), "D3": (1, 0), "D4": (1, 1)}

CH5_UNIFORM_BELIEF_COLORS = (
    "#ffe0e0",
    "#ffc9c9",
    "#ffb0b0",
    "#ef9a9a",
    "#e57373",
    "#ef5350",
    "#e53935",
    "#d32f2f",
    "#c62828",
    "#b71c1c",
    "#7f0000",
)
_CH5_UNIFORM_BELIEF_CMAP = None


def ch5_uniform_belief_z_lim(*, pad_frac=0.06) -> tuple[float, float]:
    """Shared belief-pdf color scale: max peak across D1–D4 (uniform prior)."""
    return ch5_prior_w12_z_lim("uniform", scope="global", pad_frac=pad_frac)


def _ch5_belief_pdf_cmap_t(density, *, z_lim) -> np.ndarray:
    """Map belief pdf → [0, 1] for Ch4 heatmap (high pdf = dark, low pdf = pink)."""
    from matplotlib.colors import Normalize

    lo, hi = float(z_lim[0]), float(z_lim[1])
    norm = Normalize(vmin=lo, vmax=hi)
    t = np.asarray(norm(np.asarray(density, dtype=float)), dtype=float)
    return 1.0 - t


def ch5_uniform_belief_heatmap_color(pdf, lo, hi) -> str:
    """Hex color on the inverted belief-pdf scale (likely = dark, unlikely = pink)."""
    import matplotlib as mpl

    t = float(_ch5_belief_pdf_cmap_t([float(pdf)], z_lim=(float(lo), float(hi)))[0])
    rgba = ch4_nll_heatmap_cmap()(t)
    return mpl.colors.to_hex(rgba, keep_alpha=False)


def ch5_uniform_belief_facecolors(
    density,
    *,
    z_lim=None,
    prior_kind: str = "uniform",
    surface_alpha: float | None = None,
    gamma: float | None = None,
) -> np.ndarray:
    """RGBA face colors — Ch4 heatmap; high pdf dark, low pdf pink; vmax = global peak."""
    del prior_kind, gamma
    if z_lim is None:
        z_lim = ch5_uniform_belief_z_lim()
    t = _ch5_belief_pdf_cmap_t(density, z_lim=z_lim)
    fc = ch4_nll_heatmap_cmap()(t)
    fc = np.asarray(fc, dtype=float)
    al = float(CH5_BELIEF_SURFACE_ALPHA if surface_alpha is None else surface_alpha)
    fc[..., 3] = al
    return fc


def ch5_uniform_belief_rgba_at_pdf(
    pdf,
    *,
    z_lim=None,
    gamma: float | None = None,
    alpha: float | None = None,
) -> np.ndarray:
    """RGBA for belief pdf values (trace lines, grid edges) on the shared Ch4 scale."""
    del gamma
    if z_lim is None:
        z_lim = ch5_uniform_belief_z_lim()
    t = _ch5_belief_pdf_cmap_t(pdf, z_lim=z_lim)
    fc = ch4_nll_heatmap_cmap()(t)
    fc = np.asarray(fc, dtype=float)
    al = float(CH5_BELIEF_SURFACE_ALPHA if alpha is None else alpha)
    fc[..., 3] = al
    return fc


# Legacy aliases — Ch4 heatmap replaces the custom red-only colormap.
def ch5_uniform_belief_cmap():
    return ch4_nll_heatmap_cmap()


def ch5_uniform_belief_color_t(
    density,
    *,
    z_lim=None,
    gamma: float | None = None,
) -> np.ndarray:
    """Normalized [0, 1] colormap coordinate (inverted: high pdf → 0, low pdf → 1)."""
    del gamma
    if z_lim is None:
        z_lim = ch5_uniform_belief_z_lim()
    return _ch5_belief_pdf_cmap_t(density, z_lim=z_lim)

CH5_EXPORT_DPI = CH6_EXPORT_DPI


def ch5_fig_to_image(fig, dpi=None):
    dpi = int(CH5_EXPORT_DPI if dpi is None else dpi)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def ch5_figure_grid(nrows: int, ncols: int, *, figsize=None):
    figsize = CH6_FIGSIZE if figsize is None else figsize
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.atleast_2d(axes)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.03, wspace=0.10, hspace=0.12)
    return fig, axes


def ch5_figure_grid_mixed(nrows: int, ncols: int, *, figsize=None):
    """Grid with 2D and 3D cells — returns axes[i,j] and projection flag."""
    figsize = CH6_FIGSIZE if figsize is None else figsize
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols, wspace=0.08, hspace=0.10)
    axes = np.empty((nrows, ncols), dtype=object)
    proj = np.zeros((nrows, ncols), dtype=bool)
    for i in range(nrows):
        for j in range(ncols):
            if j >= 1:
                axes[i, j] = fig.add_subplot(gs[i, j], projection="3d")
                proj[i, j] = True
            else:
                axes[i, j] = fig.add_subplot(gs[i, j])
    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.03)
    return fig, axes, proj


def ch5_crossfade_images(img_a, img_b, u: float) -> Image.Image:
    """Blend two same-size RGB images (u=0 → a, u=1 → b)."""
    u = float(np.clip(float(u), 0.0, 1.0))
    a = img_a.convert("RGB")
    b = img_b.convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size, resample=Image.Resampling.LANCZOS)
    if u <= 0.0:
        return a
    if u >= 1.0:
        return b
    return Image.blend(a, b, u)


def ch5_quadrant_zoom_frame(grid_img, row: int, col: int, zoom_u: float):
    """Smooth camera zoom into one 2×2 quadrant (shrinking crop window, no crossfade)."""
    u = float(np.clip(float(zoom_u), 0.0, 1.0))
    grid = grid_img.convert("RGB")
    w, h = grid.size
    qw, qh = w // 2, h // 2
    x0, y0 = int(col * qw), int(row * qh)
    icx, icy = w / 2.0, h / 2.0
    qcx, qcy = x0 + qw / 2.0, y0 + qh / 2.0
    cx = icx + u * (qcx - icx)
    cy = icy + u * (qcy - icy)
    cw = w * (1.0 - u) + qw * u
    ch = h * (1.0 - u) + qh * u
    left = int(round(cx - cw / 2.0))
    top = int(round(cy - ch / 2.0))
    right = int(round(cx + cw / 2.0))
    bottom = int(round(cy + ch / 2.0))
    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))
    cropped = grid.crop((left, top, right, bottom))
    return cropped.resize((w, h), resample=Image.Resampling.LANCZOS)


def ch5_dim_cell_image(
    img,
    *,
    dim_u: float = 1.0,
    grey_weight: float = 0.78,
    alpha_min: float = 0.26,
    bg=(255, 255, 255),
):
    """Desaturate + fade one quadrant cell (axes, knobs, surface, points — everything)."""
    u = float(np.clip(float(dim_u), 0.0, 1.0))
    if u <= 1e-6:
        return img.convert("RGB") if hasattr(img, "convert") else img
    rgb = img.convert("RGB")
    grey = rgb.convert("L").convert("RGB")
    gw = float(np.clip(float(grey_weight), 0.0, 1.0))
    desat = Image.blend(rgb, grey, u * gw)
    alpha = 1.0 - u * (1.0 - float(np.clip(float(alpha_min), 0.0, 1.0)))
    white = Image.new("RGB", rgb.size, tuple(int(c) for c in bg))
    return Image.blend(white, desat, alpha)


def ch5_howithink_full_rgba():
    """Full-resolution howithinkabout.png (RGBA), not the 96px badge."""
    for p in (OUTPUT_DIR / "howithinkabout.png", Path("howithinkabout.png")):
        if p.is_file():
            return Image.open(p).convert("RGBA")
    raise FileNotFoundError("howithinkabout.png not found")


def ch5_overlay_howithink_center_right(
    img,
    *,
    dim_u: float = 1.0,
    logo_u: float = 1.0,
    size_frac: float = 0.48,
    cx_frac: float = 0.72,
    cy_frac: float = 0.50,
):
    """Grey the frame and paste a large howithinkabout logo center-right."""
    base = ch5_dim_cell_image(img, dim_u=float(dim_u)).convert("RGBA")
    lu = float(np.clip(float(logo_u), 0.0, 1.0))
    if lu <= 1e-6:
        return base.convert("RGB")
    logo = ch5_howithink_full_rgba()
    w, h = base.size
    target_h = max(64, int(round(h * float(size_frac))))
    aspect = logo.width / max(logo.height, 1)
    target_w = max(64, int(round(target_h * aspect)))
    logo = logo.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
    if lu < 1.0 - 1e-6:
        a = np.asarray(logo, dtype=np.float64)
        a[..., 3] *= lu
        logo = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), mode="RGBA")
    cx = int(round(w * float(cx_frac)))
    cy = int(round(h * float(cy_frac)))
    x0 = int(cx - target_w // 2)
    y0 = int(cy - target_h // 2)
    out = base.copy()
    out.paste(logo, (x0, y0), logo)
    return out.convert("RGB")


def ch5_confetti_best_line_overlay(
    img,
    *,
    u: float,
    label: str = "best line",
    seed: int = 70,
):
    """
    Celebration overlay: popping ``label`` text + bursting confetti.

    ``u`` in [0,1]: text scales in early, confetti bursts then drifts down.
    """
    from PIL import ImageDraw, ImageFont

    u = float(np.clip(float(u), 0.0, 1.0))
    base = img.convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rng = np.random.default_rng(int(seed))

    # Text pop: ease in, slight overshoot, settle.
    if u < 0.35:
        t = u / 0.35
        scale = 0.15 + 1.05 * (t * t * (3.0 - 2.0 * t))
        text_a = t
    elif u < 0.55:
        t = (u - 0.35) / 0.20
        scale = 1.20 - 0.20 * t
        text_a = 1.0
    else:
        scale = 1.0
        text_a = 1.0

    fs = max(18, int(round(0.085 * h * scale)))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", fs)
    except Exception:
        try:
            font = ImageFont.truetype("Arial Bold.ttf", fs)
        except Exception:
            font = ImageFont.load_default()

    text = str(label)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = int(0.35 * tw), int(0.40 * th)
    box_w, box_h = tw + 2 * pad_x, th + 2 * pad_y
    bx = (w - box_w) // 2
    by = int(0.18 * h)
    # Pill background
    alpha = int(round(245 * text_a))
    draw.rounded_rectangle(
        [bx, by, bx + box_w, by + box_h],
        radius=max(8, box_h // 3),
        fill=(20, 20, 20, alpha),
        outline=(255, 255, 255, alpha),
        width=max(2, fs // 14),
    )
    draw.text(
        (bx + pad_x - bbox[0], by + pad_y - bbox[1]),
        text,
        font=font,
        fill=(255, 255, 255, alpha),
    )

    # Confetti burst from near the label, then fall.
    colors = [
        (231, 76, 60), (46, 204, 113), (52, 152, 219),
        (241, 196, 15), (155, 89, 182), (230, 126, 34),
    ]
    n_bits = 90
    burst_u = float(np.clip((u - 0.08) / 0.92, 0.0, 1.0))
    ox = w * 0.5
    oy = by + box_h * 0.55
    for i in range(n_bits):
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        speed = float(rng.uniform(0.18, 0.55)) * min(w, h)
        drift = burst_u * speed
        grav = 0.55 * (burst_u ** 2) * h
        x = ox + np.cos(ang) * drift * float(rng.uniform(0.4, 1.0))
        y = oy + np.sin(ang) * drift * 0.35 + grav * float(rng.uniform(0.6, 1.2))
        if x < -20 or y < -20 or x > w + 20 or y > h + 20:
            continue
        col = colors[i % len(colors)]
        a = int(round(230 * (1.0 - 0.35 * burst_u)))
        sz = int(rng.integers(4, 11))
        rot = float(rng.uniform(0, 360))
        # Small rotated rectangle via polygon
        hw, hh = sz * 0.7, sz * 0.35
        rad = np.deg2rad(rot + 40.0 * burst_u)
        c, s = np.cos(rad), np.sin(rad)
        corners = []
        for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
            corners.append((x + c * dx - s * dy, y + s * dx + c * dy))
        draw.polygon(corners, fill=(*col, a))

    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")


def ch5_composite_2x2_focus(
    cells,
    focus_key: str | None = None,
    *,
    focus_keys: set[str] | frozenset[str] | None = None,
    dim_u: float = 1.0,
    prev_focus: str | None = None,
    transition_u: float | None = None,
    add_lit_key: str | None = None,
    add_lit_u: float | None = None,
    grey_weight: float = 0.78,
    alpha_min: float = 0.26,
    bg=(255, 255, 255),
    gap=0,
):
    """
    2×2 grid with lit quadrant(s); inactive cells greyed/faded.

    Single-focus (legacy): pass ``focus_key``; optional ``transition_u`` /
    ``prev_focus`` animate dimming.

    Multi-focus: pass ``focus_keys`` for cells that stay bright; optional
    ``add_lit_key`` + ``add_lit_u`` brighten one more cell (others stay dimmed).
    """
    lit = set(focus_keys) if focus_keys is not None else (
        {focus_key} if focus_key is not None else set()
    )
    out = [[None, None], [None, None]]
    tu = None if transition_u is None else float(np.clip(float(transition_u), 0.0, 1.0))
    au = None if add_lit_u is None else float(np.clip(float(add_lit_u), 0.0, 1.0))
    du = float(dim_u)
    for key in CH5_DATASET_KEYS:
        i, j = _CH5_GRID_SLOTS[key]
        cell = cells[i][j]
        if cell is None:
            continue
        if focus_keys is not None or add_lit_key is not None:
            if key in lit:
                u = 0.0
            elif add_lit_key is not None and key == add_lit_key and au is not None:
                u = 1.0 - au
            else:
                u = du
        elif focus_key is None:
            u = du
        elif key == focus_key:
            u = 0.0
        elif tu is not None and prev_focus is not None and key == prev_focus:
            u = tu
        elif tu is not None and prev_focus is None:
            u = tu
        else:
            u = du
        out[i][j] = (
            cell.convert("RGB")
            if u <= 1e-6
            else ch5_dim_cell_image(
                cell, dim_u=u, grey_weight=grey_weight, alpha_min=alpha_min, bg=bg,
            )
        )
    return ch5_composite_2x2_quadrants(out, bg=bg, gap=gap)


def ch5_composite_2x2_quadrants(cells, *, bg=(255, 255, 255), gap=0):
    """Paste four equal-sized images into quadrants. ``cells`` is 2×2 [row][col]."""
    ref = next(c for row in cells for c in row if c is not None)
    cw, ch = ref.size
    gw = int(gap)
    out = Image.new("RGB", (2 * cw + gw, 2 * ch + gw), bg)
    for i in range(2):
        for j in range(2):
            im = cells[i][j]
            if im is None:
                im = Image.new("RGB", (cw, ch), (242, 242, 242))
            elif im.size != (cw, ch):
                im = im.resize((cw, ch), resample=Image.Resampling.LANCZOS)
            out.paste(im, (j * (cw + gw), i * (ch + gw)))
    return out


_CH5_BADGE = None


def ch5_howithink_badge_rgba():
    global _CH5_BADGE
    if _CH5_BADGE is not None:
        return _CH5_BADGE
    for p in (OUTPUT_DIR / "howithinkabout.png", Path("howithinkabout.png")):
        if p.is_file():
            im = Image.open(p).convert("RGBA")
            w, h = im.size
            sq = min(w, h)
            im = im.crop((0, (h - sq) // 2, sq, (h - sq) // 2 + sq))
            im = im.resize((96, 96), resample=Image.Resampling.LANCZOS)
            _CH5_BADGE = np.asarray(im, dtype=np.uint8)
            return _CH5_BADGE
    raise FileNotFoundError("howithinkabout.png not found")


def ch5_place_howithink_badge(ax, *, dx=0.02, scale=0.22):
    rgba = ch5_howithink_badge_rgba()
    icon = OffsetImage(rgba, zoom=float(scale))
    fig = ax.figure
    pos = ax.get_position()
    x = float(pos.x1) + float(dx)
    y = float(pos.y0) + 0.5 * float(pos.height)
    ab = AnnotationBbox(icon, (x, y), xycoords=fig.transFigure, frameon=False, zorder=50)
    ax.add_artist(ab)


def ch5_draw_right_text(ax, text, *, fontsize=14):
    ax.set_axis_off()
    ax.text(0.05, 0.95, str(text), transform=ax.transAxes, va="top", ha="left", fontsize=fontsize, color="#222")


def ch5_hpd_mask_from_weights(weights, mass=0.95):
    """HPD mask: smallest set of highest-probability cells covering ``mass``."""
    w = np.asarray(weights, dtype=float)
    w = w / max(float(np.sum(w)), 1e-18)
    flat = w.ravel()
    order = np.argsort(-flat)
    csum = np.cumsum(flat[order])
    k = int(np.searchsorted(csum, float(mass), side="left")) + 1
    mask = np.zeros_like(flat, dtype=bool)
    mask[order[:k]] = True
    return mask.reshape(w.shape)


def ch5_hpd_mask_2d(density, mass=0.95):
    """Boolean mask: highest-density cells covering `mass` of grid probability."""
    return ch5_hpd_mask_from_weights(density, mass=mass)


def ch5_hpd_contour_on_ax(ax, W1, W2, density, mass=0.95, *, color="#3366cc", lw=2.0, zorder=7, z_offset=0.0):
    mask = ch5_hpd_mask_2d(density, mass=mass)
    if getattr(ax, "name", "") == "3d":
        ax.contour(
            W1, W2, mask.astype(float), levels=[0.5],
            zdir="z", offset=float(z_offset),
            colors=[color], linewidths=float(lw), zorder=int(zorder),
        )
    else:
        ax.contour(
            W1, W2, mask.astype(float), levels=[0.5],
            colors=[color], linewidths=float(lw), zorder=int(zorder),
        )


def ch5_hpd_region_on_floor(
    ax3d, W1, W2, density, mass=0.95, *,
    fill_color="#2D5BFF", fill_alpha=0.32, edge_color="#08111f", edge_lw=2.0, z_offset=0.0,
):
    """Credible region on the (w_ST, w_EL) plane under the 3D landscape."""
    mask = ch5_hpd_mask_2d(density, mass=mass).astype(float)
    z0 = float(z_offset)
    ax3d.contourf(
        W1, W2, mask, levels=[0.5, 1.5],
        zdir="z", offset=z0,
        colors=[fill_color], alpha=float(fill_alpha), zorder=2,
    )
    ax3d.contour(
        W1, W2, mask, levels=[0.5],
        zdir="z", offset=z0,
        colors=[edge_color], linewidths=float(edge_lw), zorder=3,
    )


def ch5_hpd_mask_3d(weights, mass=0.95):
    """Boolean mask: highest-probability voxels covering ``mass`` of belief."""
    return ch5_hpd_mask_from_weights(weights, mass=mass)


def ch5_belief_volume_facecolors(density, *, prior_kind=None, invert=True, alpha=0.18, quantile=0.06):
    """RGBA face colors for belief voxels in (w_ST, w_EL, b) parameter space."""
    pk = str(prior_kind or "gaussian").lower()
    d = np.asarray(density, dtype=float)
    if pk == "uniform":
        z_lim = ch5_uniform_belief_z_lim()
        d = ch5_clip_belief_height(d, prior_kind=pk, z_lim=z_lim)
        fc = ch5_uniform_belief_facecolors(d, z_lim=z_lim, surface_alpha=1.0)
    else:
        lo, hi = ch5_global_density_limits(pk)
        span = max(hi - lo, 1e-12)
        normed = np.clip((d - lo) / span, 0.0, 1.0)
        t = 1.0 - normed if invert else normed
        fc = ch4_nll_heatmap_cmap()(t)
    pos = d[np.isfinite(d) & (d > 0)]
    thresh = float(np.quantile(pos, quantile)) if pos.size else 0.0
    fc = np.asarray(fc, dtype=float)
    fc[..., 3] = float(alpha) * (d >= thresh)
    return fc


def ch5_draw_belief_volume(ax3d, w1_edges, w2_edges, b_edges, density, **kw):
    """Ghost voxels: belief mass as color at each (w_ST, w_EL, b) cell."""
    d = np.asarray(density, dtype=float)
    fc = ch5_belief_volume_facecolors(d, **kw)
    visible = fc[..., 3] > 1e-6
    if not np.any(visible):
        return
    ax3d.voxels(
        np.asarray(w1_edges, dtype=float),
        np.asarray(w2_edges, dtype=float),
        np.asarray(b_edges, dtype=float),
        visible,
        facecolors=fc,
        edgecolor=(0, 0, 0, 0),
        linewidth=0.0,
        shade=False,
    )


def ch5_draw_map_parameter_marker(
    ax3d, ws, we, bb, *, color="#e8b020", edgecolor="white", s=260.0,
    as_sphere=False, sphere_radius=0.10,
):
    """MAP point in full (w_ST, w_EL, b) parameter space.

    ``as_sphere=True`` draws a tiny ``plot_surface`` ball so the marker shares
    mplot3d's surface projector with Laplace ellipsoids (scatter often drifts).
    """
    if bool(as_sphere):
        o = np.array([float(ws), float(we), float(bb)], dtype=np.float64)
        r = float(sphere_radius)
        u = np.linspace(0.0, 2.0 * np.pi, 14, endpoint=True)
        v = np.linspace(0.0, np.pi, 10, endpoint=True)
        uu, vv = np.meshgrid(u, v)
        X = o[0] + r * np.cos(uu) * np.sin(vv)
        Y = o[1] + r * np.sin(uu) * np.sin(vv)
        Z = o[2] + r * np.cos(vv)
        ax3d.plot_surface(
            X, Y, Z,
            color=str(color),
            edgecolor=str(edgecolor),
            linewidth=0.35,
            alpha=1.0,
            shade=False,
            zorder=50,
        )
        return
    ax3d.scatter(
        [float(ws)], [float(we)], [float(bb)],
        color=color, edgecolors=edgecolor, linewidths=2.0,
        s=float(s), depthshade=False, zorder=40,
    )


def ch5_draw_map_parameter_annotation(
    fig, ax3d, ws, we, bb, *, annotate_fn, color=None, edgecolor="white",
    text_color="white", label="most plausible line",
):
    """Arrow + label pointing at MAP in parameter space (uses Ch4 ``annotate_fn``)."""
    if annotate_fn is None:
        return
    annotate_fn(
        fig, ax3d, float(ws), float(we), float(bb),
        label=str(label),
        color=color, edgecolor=edgecolor, text_color=text_color,
    )


def ch5_marginal_credible_intervals(w1_axis, w2_axis, b_axis, mask):
    """1D credible intervals from a 3D HPD mask on (w_ST, w_EL, b)."""
    m = np.asarray(mask, dtype=bool)
    w1_axis = np.asarray(w1_axis, dtype=float)
    w2_axis = np.asarray(w2_axis, dtype=float)
    b_axis = np.asarray(b_axis, dtype=float)
    out = {}
    st = w1_axis[m.any(axis=(1, 2))]
    el = w2_axis[m.any(axis=(0, 2))]
    bb = b_axis[m.any(axis=(0, 1))]
    out["st"] = (float(st.min()), float(st.max())) if st.size else (np.nan, np.nan)
    out["el"] = (float(el.min()), float(el.max())) if el.size else (np.nan, np.nan)
    out["b"] = (float(bb.min()), float(bb.max())) if bb.size else (np.nan, np.nan)
    return out


def ch5_hpd_radial_distances(w1_axis, w2_axis, b_axis, ws, we, bb):
    """Euclidean distance from each grid cell center to MAP in (w_ST, w_EL, b)."""
    w1 = np.asarray(w1_axis, dtype=float)
    w2 = np.asarray(w2_axis, dtype=float)
    b = np.asarray(b_axis, dtype=float)
    W1, W2, B = np.meshgrid(w1, w2, b, indexing="ij")
    dist = np.sqrt(
        (W1 - float(ws)) ** 2 + (W2 - float(we)) ** 2 + (B - float(bb)) ** 2,
    )
    return W1, W2, B, dist


def ch5_belief_gradient_rgba(density, *, prior_kind=None, invert=True, alpha=0.72, lo=None, hi=None):
    """Ch4 heatmap RGBA for belief density (high ρ → dark when ``invert=True``)."""
    if lo is None or hi is None:
        lo, hi = ch5_global_density_limits(prior_kind or "gaussian")
    d = np.asarray(density, dtype=float)
    span = max(float(hi) - float(lo), 1e-12)
    normed = np.clip((d - float(lo)) / span, 0.0, 1.0)
    t = 1.0 - normed if invert else normed
    rgba = np.asarray(ch4_nll_heatmap_cmap()(t), dtype=float)
    if rgba.ndim == 0:
        rgba = rgba.reshape(1, 4)
    elif rgba.ndim == 1:
        rgba = rgba.reshape(1, 4)
    rgba[..., 3] = float(alpha)
    return rgba


def ch5_draw_hpd_point_cloud(
    ax3d,
    w1_axis,
    w2_axis,
    b_axis,
    mask,
    density,
    *,
    ws,
    we,
    bb,
    fill_u=1.0,
    alpha=0.72,
    size=36.0,
    prior_kind=None,
):
    """Credible region as a belief-colored point cloud, growing outward from MAP."""
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return
    W1, W2, B, dist = ch5_hpd_radial_distances(w1_axis, w2_axis, b_axis, ws, we, bb)
    max_d = float(np.max(dist[m]))
    u = float(np.clip(fill_u, 0.0, 1.0))
    if max_d <= 1e-12:
        revealed = m
    else:
        revealed = m & (dist <= u * max_d + 1e-12)
    if not np.any(revealed):
        return
    d = np.asarray(density, dtype=float)
    colors = ch5_belief_gradient_rgba(d[revealed], alpha=alpha, prior_kind=prior_kind)
    ax3d.scatter(
        W1[revealed], W2[revealed], B[revealed],
        c=colors, s=float(size),
        depthshade=False, edgecolors="none", marker="o", zorder=20,
    )


def ch5_draw_hpd_voxels(ax3d, w1_edges, w2_edges, b_edges, mask, *, face_rgba=(0.20, 0.35, 0.85, 0.78)):
    """Draw HPD voxels in full (w_ST, w_EL, b) parameter space."""
    filled = np.asarray(mask, dtype=bool)
    if not np.any(filled):
        return
    fc = np.empty(filled.shape + (4,), dtype=float)
    fc[:] = np.asarray(face_rgba, dtype=float)
    ax3d.voxels(
        np.asarray(w1_edges, dtype=float),
        np.asarray(w2_edges, dtype=float),
        np.asarray(b_edges, dtype=float),
        filled,
        facecolors=fc,
        edgecolor="#08111f",
        linewidth=0.2,
        shade=False,
    )


def ch5_hpd_voxel_diag_projection(w1_centers, w2_centers, b_centers, *, bounds):
    """Diagonal sweep coordinate for voxel-fill reveal (corner → corner)."""
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = bounds
    start = np.array([dlo1, dlo2, dlob], dtype=np.float64)
    end = np.array([dhi1, dhi2, dhib], dtype=np.float64)
    diag = end - start
    max_proj = float(np.dot(diag, diag))
    W1, W2, B = np.meshgrid(w1_centers, w2_centers, b_centers, indexing="ij")
    proj = (W1 - start[0]) * diag[0] + (W2 - start[1]) * diag[1] + (B - start[2]) * diag[2]
    return proj, max_proj


def ch5_draw_hpd_voxels_fill(
    ax3d,
    w1_axis,
    w2_axis,
    b_axis,
    hpd_mask,
    density,
    ws,
    we,
    bb,
    sweep_u,
    *,
    prior_kind=None,
    **kw,
):
    """Reveal HPD point cloud from MAP outward (ellipsoid-like growth)."""
    ch5_draw_hpd_point_cloud(
        ax3d, w1_axis, w2_axis, b_axis, hpd_mask, density,
        ws=ws, we=we, bb=bb, fill_u=sweep_u, prior_kind=prior_kind, **kw,
    )


def _ch5_hpd_cell_centers(w1_axis, w2_axis, b_axis, mask):
    w1 = np.asarray(w1_axis, dtype=float)
    w2 = np.asarray(w2_axis, dtype=float)
    b = np.asarray(b_axis, dtype=float)
    W1, W2, B = np.meshgrid(w1, w2, b, indexing="ij")
    m = np.asarray(mask, dtype=bool)
    return W1[m], W2[m], B[m]


def ch5_draw_hpd_orthogonal_projection(
    ax3d,
    w1_axis,
    w2_axis,
    b_axis,
    mask,
    intervals,
    ws,
    we,
    bb,
    bounds,
    axis_key,
    *,
    shadow_u=0.0,
    collapse_u=0.0,
    color="#3366cc",
    shadow_alpha=0.42,
    shadow_size=22.0,
    line_lw=5.0,
    line_alpha=0.92,
    map_shadow_color="#e04a4a",
    map_shadow_alpha=0.58,
    map_shadow_size=54.0,
    line_ws=None,
    line_we=None,
    line_bb=None,
):
    """Orthogonal voxel shadow on a box face, then collapse to a marginal interval.

    Faces (Ch5 view bounds) — shadow is orthogonal onto the face; interval axis ⊥ the face normal:
      el — plane w_EL = +3; shadow in (w_ST, b); interval along w_ST
      st — plane w_ST = +3; shadow in (w_EL, b); interval along b
      b  — plane b = −3; shadow in (w_ST, w_EL); interval along w_EL

    ``ws, we, bb`` locate the on-interval marker (probe / MAP). Optional
    ``line_ws/we/bb`` pin the interval segment so it does not slide with the probe.
    """
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = bounds
    w1c, w2c, bc = _ch5_hpd_cell_centers(w1_axis, w2_axis, b_axis, mask)
    if w1c.size == 0:
        return

    su = float(np.clip(shadow_u, 0.0, 1.0))
    cu = float(np.clip(collapse_u, 0.0, 1.0))
    key = str(axis_key).lower()
    ws, we, bb = float(ws), float(we), float(bb)
    # Interval geometry: pin to explicit anchors when given (wander clips).
    lws = float(ws if line_ws is None else line_ws)
    lwe = float(we if line_we is None else line_we)
    lbb = float(bb if line_bb is None else line_bb)

    if key == "el":
        plane = float(dhi2)
        lo, hi = intervals["st"]
        sx, sy, sz = w1c, np.full_like(w1c, plane), bc
        tx = np.clip(w1c, float(lo), float(hi))
        ty = np.full_like(w1c, plane)
        tz = np.full_like(w1c, lbb)
        lx = np.array([float(lo), float(hi)], dtype=float)
        ly = np.array([plane, plane], dtype=float)
        lz = np.array([lbb, lbb], dtype=float)
        mx, my, mz = float(np.clip(ws, float(lo), float(hi))), plane, lbb
    elif key == "st":
        plane = float(dhi1)
        lo, hi = intervals["b"]
        sx, sy, sz = np.full_like(w2c, plane), w2c, bc
        tx = np.full_like(w2c, plane)
        ty = np.full_like(w2c, lwe)
        tz = np.clip(bc, float(lo), float(hi))
        lx = np.array([plane, plane], dtype=float)
        ly = np.array([lwe, lwe], dtype=float)
        lz = np.array([float(lo), float(hi)], dtype=float)
        mx, my, mz = plane, lwe, float(np.clip(bb, float(lo), float(hi)))
    elif key == "b":
        plane = float(dlob)
        lo, hi = intervals["el"]
        sx, sy, sz = w1c, w2c, np.full_like(bc, plane)
        tx = np.full_like(w1c, lws)
        ty = np.clip(w2c, float(lo), float(hi))
        tz = np.full_like(bc, plane)
        lx = np.array([lws, lws], dtype=float)
        ly = np.array([float(lo), float(hi)], dtype=float)
        lz = np.array([plane, plane], dtype=float)
        mx, my, mz = lws, float(np.clip(we, float(lo), float(hi))), plane
    else:
        return

    if not (np.isfinite(lo) and np.isfinite(hi)):
        return

    if su > 1e-6 and cu < 1.0 - 1e-6:
        px = (1.0 - cu) * sx + cu * tx
        py = (1.0 - cu) * sy + cu * ty
        pz = (1.0 - cu) * sz + cu * tz
        ax3d.scatter(
            px, py, pz,
            c=str(color), s=float(shadow_size), alpha=float(shadow_alpha) * su,
            depthshade=False, edgecolors="none", marker="o", zorder=18,
        )

    if cu > 1.0 - 1e-6:
        ax3d.plot(
            lx, ly, lz,
            color=str(color), lw=float(line_lw), alpha=float(line_alpha),
            solid_capstyle="round", zorder=22,
        )

    if su > 1e-6 or cu > 1e-6:
        m_alpha = float(map_shadow_alpha) * max(float(su), 1.0 if cu > 1e-6 else 0.0)
        ax3d.scatter(
            [mx], [my], [mz],
            c=str(map_shadow_color), s=float(map_shadow_size), alpha=m_alpha,
            depthshade=False, edgecolors="white", linewidths=1.2, marker="o", zorder=24,
        )


def ch5_map_basis_directions():
    """Unit directions from MAP: ẑ, (w_ST, w_EL)=(1,−1), and their cross product."""
    u_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u_diag = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    u_diag /= float(np.linalg.norm(u_diag))
    u_ortho = np.cross(u_z, u_diag)
    n = float(np.linalg.norm(u_ortho))
    if n < 1e-12:
        u_ortho = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        u_ortho /= float(np.linalg.norm(u_ortho))
    else:
        u_ortho /= n
    return (u_z, u_diag, u_ortho)


def ch5_blend_param_colors(weights, colors=None):
    """Blend ST/EL/b accent colors by non-negative weights (e.g. |eigenvector|)."""
    from matplotlib.colors import to_rgb

    if colors is None:
        from ch4_layout import CH4_GD_PARTIAL_COLORS
        colors = CH4_GD_PARTIAL_COLORS
    w = np.asarray(weights, dtype=np.float64).ravel()[:3]
    w = np.abs(w)
    s = float(np.sum(w))
    if s < 1e-12:
        w = np.ones(3, dtype=np.float64) / 3.0
    else:
        w = w / s
    rgb = np.zeros(3, dtype=np.float64)
    for i in range(3):
        rgb += w[i] * np.asarray(to_rgb(str(colors[i])), dtype=np.float64)
    return "#{:02x}{:02x}{:02x}".format(
        int(np.clip(rgb[0], 0, 1) * 255 + 0.5),
        int(np.clip(rgb[1], 0, 1) * 255 + 0.5),
        int(np.clip(rgb[2], 0, 1) * 255 + 0.5),
    )


def ch5_eigen_quiver_color_plan(dirs, *, dominance=0.72, black="#111111", colors=None):
    """Color an eigen-arrow only when one ST/EL/b axis clearly dominates.

    Returns ``(quiver_colors, axis_active)`` where ``axis_active[i]`` is True iff
    some quiver is dominated by parameter ``i`` (and thus worth coloring in H).
    Mixed directions stay ``black`` and do not light up Hessian cells.
    """
    if colors is None:
        from ch4_layout import CH4_GD_PARTIAL_COLORS
        colors = CH4_GD_PARTIAL_COLORS
    q_cols = []
    axis_active = [False, False, False]
    dom = float(dominance)
    for u in dirs:
        w = np.abs(np.asarray(u, dtype=np.float64).ravel()[:3])
        s = float(np.sum(w))
        if s < 1e-12:
            q_cols.append(str(black))
            continue
        w = w / s
        k = int(np.argmax(w))
        if float(w[k]) >= dom:
            q_cols.append(str(colors[k]))
            axis_active[k] = True
        else:
            q_cols.append(str(black))
    return q_cols, axis_active


def ch5_eigen_quiver_newton_colors(dirs, *, colors=None):
    """Unique ST/EL/b Newton accents (blue / orange / green) for each eigen-arrow.

    Greedy one-to-one matching by |component| so mixed ST–EL directions still
    recall Chapter 4's Newton's-method partial colors instead of staying black.
    """
    if colors is None:
        from ch4_layout import CH4_GD_PARTIAL_COLORS
        colors = CH4_GD_PARTIAL_COLORS
    dirs_arr = [np.asarray(u, dtype=np.float64).ravel()[:3] for u in list(dirs)[:3]]
    while len(dirs_arr) < 3:
        dirs_arr.append(np.zeros(3, dtype=np.float64))
    W = np.abs(np.stack(dirs_arr, axis=0))
    row_sum = np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    W = W / row_sum
    assigned = [None, None, None]
    used_ax: set[int] = set()
    cands = sorted(
        ((float(W[q, a]), q, a) for q in range(3) for a in range(3)),
        reverse=True,
    )
    for _score, q, a in cands:
        if assigned[q] is not None or a in used_ax:
            continue
        assigned[q] = a
        used_ax.add(a)
        if len(used_ax) >= 3:
            break
    for q in range(3):
        if assigned[q] is None:
            for a in range(3):
                if a not in used_ax:
                    assigned[q] = a
                    used_ax.add(a)
                    break
            if assigned[q] is None:
                assigned[q] = q
    return [str(colors[int(a)]) for a in assigned]


def ch5_hessian_cell_colors_for_axes(axis_active, *, colors=None, black=None):
    """Color H cells only for axes that have a matching colored quiver.

    Diagonal ``H_ii`` gets the axis accent when ``axis_active[i]``; everything
    else stays ``black`` / ``None`` (rendered as default ink).
    """
    if colors is None:
        from ch4_layout import CH4_GD_PARTIAL_COLORS
        colors = CH4_GD_PARTIAL_COLORS
    active = [bool(x) for x in list(axis_active)[:3]]
    while len(active) < 3:
        active.append(False)
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            if i == j and active[i]:
                row.append(str(colors[i]))
            else:
                row.append(black)
        out.append(row)
    return out


def ch5_hessian_cell_colors(colors=None):
    """3×3 cell colors matching Hessian ∂²NLL / ∂θᵢ∂θⱼ (ST / EL / b accents)."""
    if colors is None:
        from ch4_layout import CH4_GD_PARTIAL_COLORS
        colors = CH4_GD_PARTIAL_COLORS
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            if i == j:
                row.append(str(colors[i]))
            else:
                row.append(ch5_blend_param_colors([1.0 if k in (i, j) else 0.0 for k in range(3)], colors))
        out.append(row)
    return out


def ch5_draw_map_basis_quivers(
    ax3d,
    ws,
    we,
    bb,
    *,
    length=1.48,
    lengths=None,
    dirs=None,
    lw=5.2,
    colors=("#111111", "#111111", "#111111"),
    alpha=0.98,
):
    """Thick directional quivers at the plausible point (geometric triad or eigen dirs).

    Uses ``plot`` line segments (not ``Axes3D.quiver``) so the shaft base sits on
    the MAP marker — mplot3d's quiver often starts short of ``(ws, we, bb)``.
    """
    use_dirs = list(dirs) if dirs is not None else list(ch5_map_basis_directions())
    cols = list(colors) + ["#333333"] * 3
    if lengths is None:
        lens = [float(length)] * 3
    else:
        lens = [float(x) for x in lengths]
        while len(lens) < 3:
            lens.append(float(length))
    origin = np.array([float(ws), float(we), float(bb)], dtype=np.float64)
    for i, u in enumerate(use_dirs[:3]):
        u = np.asarray(u, dtype=np.float64).ravel()[:3]
        n = float(np.linalg.norm(u))
        if n < 1e-12:
            continue
        u = u / n
        L = float(lens[i])
        tip = origin + L * u
        # Shaft.
        ax3d.plot(
            [origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]],
            color=str(cols[i]),
            linewidth=float(lw),
            solid_capstyle="round",
            alpha=float(alpha),
            zorder=30,
        )
        # Simple arrowhead (short wider stub near the tip).
        head = max(0.12 * L, 0.08)
        base = tip - head * u
        ax3d.plot(
            [base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]],
            color=str(cols[i]),
            linewidth=float(lw) * 1.55,
            solid_capstyle="round",
            alpha=float(alpha),
            zorder=31,
        )


def ch5_laplace_ellipsoid_radii(evals, lengths, *, scale=1.0):
    """Semi-axis lengths for a Laplace level set, matching quiver length ratios."""
    ev = np.clip(np.asarray(evals, dtype=np.float64).ravel()[:3], 1e-12, None)
    lens = np.asarray(lengths, dtype=np.float64).ravel()[:3]
    # Match quiver proportions (already ∝ 1/√λ and on-screen clamped).
    if lens.size >= 3 and float(np.max(lens)) > 1e-9:
        return float(scale) * lens
    inv = 1.0 / np.sqrt(ev)
    return float(scale) * inv / float(np.max(inv))


def ch5_draw_laplace_ellipsoid(
    ax3d,
    origin,
    dirs,
    radii,
    *,
    n_u=18,
    n_v=12,
    face_color="#5b8def",
    face_alpha=0.16,
    edge_color="#2a4a7a",
    edge_alpha=0.35,
    edge_lw=0.55,
    zorder=12,
):
    """Translucent quadratic level-set ellipsoid at the MAP (Laplace local curvature)."""
    o = np.asarray(origin, dtype=np.float64).ravel()[:3]
    R = np.column_stack([np.asarray(d, dtype=np.float64).ravel()[:3] for d in dirs[:3]])
    rad = np.asarray(radii, dtype=np.float64).ravel()[:3]
    # Unit sphere → ellipsoid in eigen basis.
    u = np.linspace(0.0, 2.0 * np.pi, int(n_u), endpoint=True)
    v = np.linspace(0.0, np.pi, int(n_v), endpoint=True)
    uu, vv = np.meshgrid(u, v)
    x0 = np.cos(uu) * np.sin(vv)
    y0 = np.sin(uu) * np.sin(vv)
    z0 = np.cos(vv)
    pts = np.stack([x0.ravel(), y0.ravel(), z0.ravel()], axis=0)
    pts = (R @ (rad[:, None] * pts))
    X = o[0] + pts[0].reshape(x0.shape)
    Y = o[1] + pts[1].reshape(x0.shape)
    Z = o[2] + pts[2].reshape(x0.shape)
    ax3d.plot_surface(
        X, Y, Z,
        color=str(face_color),
        alpha=float(face_alpha),
        linewidth=0.0,
        antialiased=True,
        shade=False,
        zorder=int(zorder),
    )
    if float(edge_alpha) > 1e-4:
        ax3d.plot_wireframe(
            X, Y, Z,
            color=str(edge_color),
            alpha=float(edge_alpha),
            linewidth=float(edge_lw),
            rstride=max(1, int(n_v) // 4),
            cstride=max(1, int(n_u) // 6),
            zorder=int(zorder) + 1,
        )


def ch5_draw_axis_credible_projection(
    ax3d,
    axis,
    interval,
    bounds,
    *,
    color="#3366cc",
    alpha=0.72,
    lw=4.0,
    zorder=10,
):
    """Highlight marginal credible interval on a parameter axis."""
    lo, hi = interval
    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = bounds
    al = float(np.clip(alpha, 0.0, 1.0))
    c = str(color)
    if axis == "st":
        ax3d.plot([lo, hi], [dlo2, dlo2], [dlob, dlob], color=c, lw=lw, alpha=al, zorder=zorder)
        ax3d.plot([lo, hi], [dhi2, dhi2], [dhib, dhib], color=c, lw=lw, alpha=al, zorder=zorder)
        ax3d.plot([lo, lo], [dlo2, dhi2], [dlob, dlob], color=c, lw=lw * 0.65, alpha=al * 0.55, zorder=zorder - 1)
        ax3d.plot([hi, hi], [dlo2, dhi2], [dlob, dlob], color=c, lw=lw * 0.65, alpha=al * 0.55, zorder=zorder - 1)
    elif axis == "el":
        ax3d.plot([dlo1, dlo1], [lo, hi], [dlob, dlob], color=c, lw=lw, alpha=al, zorder=zorder)
        ax3d.plot([dhi1, dhi1], [lo, hi], [dhib, dhib], color=c, lw=lw, alpha=al, zorder=zorder)
        ax3d.plot([dlo1, dhi1], [lo, lo], [dlob, dlob], color=c, lw=lw * 0.65, alpha=al * 0.55, zorder=zorder - 1)
        ax3d.plot([dlo1, dhi1], [hi, hi], [dlob, dlob], color=c, lw=lw * 0.65, alpha=al * 0.55, zorder=zorder - 1)
    else:
        ax3d.plot([dlo1, dlo1], [dlo2, dlo2], [lo, hi], color=c, lw=lw, alpha=al, zorder=zorder)
        ax3d.plot([dhi1, dhi1], [dhi2, dhi2], [lo, hi], color=c, lw=lw, alpha=al, zorder=zorder)
        ax3d.plot([dlo1, dlo1], [dlo2, dhi2], [lo, lo], color=c, lw=lw * 0.65, alpha=al * 0.55, zorder=zorder - 1)
        ax3d.plot([dlo1, dlo1], [dlo2, dhi2], [hi, hi], color=c, lw=lw * 0.65, alpha=al * 0.55, zorder=zorder - 1)


def ch5_draw_interval_parallelepiped(
    ax3d,
    intervals,
    *,
    reveal_u=1.0,
    grow_from=None,
    face_rgba=(0.20, 0.40, 0.80, 0.30),
    edge_color="#3366cc",
    edge_alpha=0.88,
    edge_lw=1.9,
):
    """Axis-aligned box from the product of three marginal credible intervals.

    ``reveal_u`` grows the box from ``grow_from`` (MAP) toward the full interval
    extents; alpha scales with reveal. Color matches HPD projection blue.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    u = float(np.clip(reveal_u, 0.0, 1.0))
    if u < 1e-4:
        return
    st = intervals.get("st")
    el = intervals.get("el")
    bb = intervals.get("b")
    if st is None or el is None or bb is None:
        return
    st_lo, st_hi = float(st[0]), float(st[1])
    el_lo, el_hi = float(el[0]), float(el[1])
    b_lo, b_hi = float(bb[0]), float(bb[1])
    if not all(np.isfinite(v) for v in (st_lo, st_hi, el_lo, el_hi, b_lo, b_hi)):
        return

    if grow_from is None:
        cx = 0.5 * (st_lo + st_hi)
        cy = 0.5 * (el_lo + el_hi)
        cz = 0.5 * (b_lo + b_hi)
    else:
        cx, cy, cz = (float(grow_from[0]), float(grow_from[1]), float(grow_from[2]))

    def _lerp(a, b):
        return (1.0 - u) * a + u * b

    x0, x1 = _lerp(cx, st_lo), _lerp(cx, st_hi)
    y0, y1 = _lerp(cy, el_lo), _lerp(cy, el_hi)
    z0, z1 = _lerp(cz, b_lo), _lerp(cz, b_hi)
    if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9 and abs(z1 - z0) < 1e-9:
        return

    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=float)
    faces = [
        [corners[i] for i in idxs]
        for idxs in (
            (0, 1, 2, 3), (4, 5, 6, 7),
            (0, 1, 5, 4), (2, 3, 7, 6),
            (0, 3, 7, 4), (1, 2, 6, 5),
        )
    ]
    fr = np.asarray(face_rgba, dtype=float).reshape(4).copy()
    fr[3] = float(np.clip(fr[3] * u, 0.0, 1.0))
    ax3d.add_collection3d(Poly3DCollection(
        faces,
        facecolors=[tuple(fr)] * len(faces),
        edgecolors="none",
        linewidths=0.0,
        shade=False,
        zorder=16,
    ))
    # 12 edges
    edge_idx = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    ea = float(np.clip(edge_alpha * u, 0.0, 1.0))
    for i, j in edge_idx:
        p, q = corners[i], corners[j]
        ax3d.plot(
            [p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
            color=str(edge_color), lw=float(edge_lw), alpha=ea,
            solid_capstyle="round", zorder=17,
        )


def ch5_draw_landscape_origin_guides(ax3d, *, guides: dict) -> None:
    """Emphasize w_EL=0 / w_ST=0 ticks and black floor guides toward parameter origin."""
    if not guides:
        return
    alpha = float(guides.get("lines_alpha", 1.0))
    if alpha < 1e-3:
        return

    w1_lo = float(guides["w1_lo"])
    w1_hi = float(guides["w1_hi"])
    w2_lo = float(guides["w2_lo"])
    w2_hi = float(guides["w2_hi"])
    z_floor = float(guides["z_floor"])
    map_z = float(guides.get("map_z", z_floor))

    el_tick_u = float(np.clip(float(guides.get("el_tick_u", 0.0)), 0.0, 1.0))
    st_tick_u = float(np.clip(float(guides.get("st_tick_u", 0.0)), 0.0, 1.0))
    el_floor_u = float(np.clip(float(guides.get("el_floor_u", 0.0)), 0.0, 1.0))
    st_floor_u = float(np.clip(float(guides.get("st_floor_u", 0.0)), 0.0, 1.0))
    belief_u = float(np.clip(float(guides.get("belief_axis_u", 0.0)), 0.0, 1.0))

    col = "black"
    al = float(np.clip(alpha, 0.0, 1.0))
    lw_line = 2.6
    tick_len0, tick_len1 = 0.18, 0.62
    tick_lw0, tick_lw1 = 3.2, 11.0
    span1 = max(abs(w1_hi - w1_lo), 1e-6)
    span2 = max(abs(w2_hi - w2_lo), 1e-6)

    if el_tick_u > 1e-4:
        tlen = (tick_len0 + (tick_len1 - tick_len0) * el_tick_u) * span2
        tlw = tick_lw0 + (tick_lw1 - tick_lw0) * el_tick_u
        ax3d.plot(
            [w1_lo, w1_lo], [-0.5 * tlen, 0.5 * tlen], [z_floor, z_floor],
            color=col, linewidth=tlw, alpha=al, zorder=55,
        )

    if st_tick_u > 1e-4:
        tlen = (tick_len0 + (tick_len1 - tick_len0) * st_tick_u) * span1
        tlw = tick_lw0 + (tick_lw1 - tick_lw0) * st_tick_u
        ax3d.plot(
            [-0.5 * tlen, 0.5 * tlen], [w2_lo, w2_lo], [z_floor, z_floor],
            color=col, linewidth=tlw, alpha=al, zorder=55,
        )

    if el_floor_u > 1e-4:
        w1_end = float(w1_lo + (0.0 - w1_lo) * el_floor_u)
        ax3d.plot(
            [w1_lo, w1_end], [0.0, 0.0], [z_floor, z_floor],
            color=col, linewidth=lw_line, alpha=al, zorder=54,
        )

    if st_floor_u > 1e-4:
        w2_end = float(w2_lo + (0.0 - w2_lo) * st_floor_u)
        ax3d.plot(
            [0.0, 0.0], [w2_lo, w2_end], [z_floor, z_floor],
            color=col, linewidth=lw_line, alpha=al, zorder=54,
        )

    if belief_u > 1e-4:
        z_top = z_floor + (map_z - z_floor) * belief_u
        ax3d.plot(
            [0.0, 0.0], [0.0, 0.0], [z_floor, z_top],
            color=col, linewidth=lw_line + 0.8, alpha=al, zorder=56,
        )


def ch5_draw_zero_axis_cross_2d(ax, *, u: float, xlim, ylim) -> None:
    """Distinct ST=0 / EL=0 lines on the 2D panel (no labels)."""
    al = float(np.clip(float(u), 0.0, 1.0))
    if al < 1e-4:
        return
    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    kw = dict(color="#1a1a1a", linewidth=2.4, alpha=al, zorder=3.5, solid_capstyle="butt")
    if x0 <= 0.0 <= x1:
        ax.plot([0.0, 0.0], [y0, y1], **kw)
    if y0 <= 0.0 <= y1:
        ax.plot([x0, x1], [0.0, 0.0], **kw)


def ch5_draw_zero_axis_cross_3d(ax3d, *, u: float, w1_lo, w1_hi, w2_lo, w2_hi, z_floor) -> None:
    """w_ST=0 and w_EL=0 lines on the belief floor plane."""
    al = float(np.clip(float(u), 0.0, 1.0))
    if al < 1e-4:
        return
    kw = dict(color="#1a1a1a", linewidth=2.6, alpha=al, zorder=52)
    # w_EL = 0: vary w_ST
    ax3d.plot(
        [float(w1_lo), float(w1_hi)], [0.0, 0.0], [float(z_floor), float(z_floor)], **kw,
    )
    # w_ST = 0: vary w_EL
    ax3d.plot(
        [0.0, 0.0], [float(w2_lo), float(w2_hi)], [float(z_floor), float(z_floor)], **kw,
    )
