"""Chapter 9 — support vector machines, in the logistic-regression picture.

Each clip's first frame is the previous clip's last frame. Students are the
Chapter 1 checks and crosses. The constrained problem is the Chapter 4/6
voxel lattice: three knob axes, color = the objective.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from hita.config.profile import active_profile
from hita.export.spec import FrameSpec
from hita.primitives.colormap import CMAP
from hita.primitives.math import lerp, sigmoid, smoothstep
from hita.primitives.svm import (
    CORE_FAIL,
    CORE_PASS,
    LEFT_PT,
    MID_PT,
    MOVED_ALPHA,
    MOVED_B,
    MOVED_W,
    NOISE_FAIL,
    NOISE_PASS,
    OPT_B,
    OPT_W,
    PAIR_ALPHA,
    PAIR_B,
    PAIR_W,
    RIGHT_PT,
    SIMPLE_BOUNDS,
    cast_rows,
    logistic_loss,
    moon_set,
    separator_at_angle,
    signed_scores,
    small_logistic_fit,
    small_soft_table,
    smo_curve,
    voxel_lattice,
)

FIGSIZE = (15.0, 9.5)
PASS_COLOR = "#2ca02c"
FAIL_COLOR = "#d62728"
LINE_COLOR = "#1a1a1a"
EDGE_COLOR = "#5c6b7a"
STRIP_COLOR = "#b7c9dc"
NAVY = "#1f4e79"
GOLD = "#e8b020"
HINGE_COLOR = "#1f4e79"
LOGIT_COLOR = "#c0392b"

SCATTER = [0.10, 0.145, 0.82, 0.74]
DUO_L = [0.055, 0.15, 0.48, 0.72]
DUO_R = [0.575, 0.15, 0.38, 0.70]

FULL_ZOOM = 0.20
TRIO_ZOOM = 0.115
FONT = 13.75
AXIS_FONT = 15.0


def _f(v: float) -> float:
    return round(float(v), 6)


def _idx(spec: dict, key: str, default: int = -1) -> int:
    """Index fields may be 0. ``or -1`` would treat that as missing."""
    if key not in spec or spec[key] is None:
        return int(default)
    return int(spec[key])


def _N(hq: int, draft: int | None = None) -> int:
    return max(2, active_profile().scale_motion(hq, draft if draft is not None else max(2, hq // 5)))


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------
def build_pack_ch9(clip_id: str, ctx: dict[str, Any]) -> dict[str, Any]:
    profile = active_profile()
    from hita.primitives.svm import _pack_points

    cast = _pack_points(cast_rows())
    moons = moon_set()
    # Cubic cells. Draft stays coarse; HQ is a fine landscape, like the logistic voxels.
    cell = 0.22 if profile.is_draft() else 0.082
    spans = (
        SIMPLE_BOUNDS[1] - SIMPLE_BOUNDS[0],
        SIMPLE_BOUNDS[3] - SIMPLE_BOUNDS[2],
        SIMPLE_BOUNDS[5] - SIMPLE_BOUNDS[4],
    )
    n_vox = tuple(max(6, int(round(span / cell))) for span in spans)
    core = np.array([CORE_FAIL, CORE_PASS], dtype=int)
    tight = np.array([CORE_FAIL, MID_PT], dtype=int)

    def _take(idx):
        return np.column_stack([cast["x"][idx], cast["e"][idx]]), cast["ypm"][idx]

    X2, y2 = _take(core)
    X3, y3 = _take(tight)
    vox = voxel_lattice(n_vox, X2, y2, bounds=SIMPLE_BOUNDS)
    vox_tight = voxel_lattice(n_vox, X3, y3, bounds=SIMPLE_BOUNDS)
    vox["min_yf_tight"] = np.asarray(vox_tight["min_yf"], dtype=np.float64)
    from hita.primitives.icons import load_icon_arrays

    check, cross = load_icon_arrays()
    Cs, wb = small_soft_table()
    return {
        "clip_id": clip_id,
        "figsize": FIGSIZE,
        "dpi": int(profile.export_dpi),
        "font_size": FONT,
        "axis_label_size": AXIS_FONT,
        "check": check,
        "cross": cross,
        "sep": cast,
        "curve": moons,
        "voxel": vox,
        "core_idx": core,
        "tight_idx": tight,
        "soft_C": Cs,
        "soft_wb": wb,
        "logit_w": small_logistic_fit(),
        "grid_n": 40 if profile.is_draft() else 90,
        "loss_n": 80 if profile.is_draft() else 160,
        "n_students": int(len(cast["x"])),
    }



# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------
class _Plan:
    def __init__(self, clip_id: str):
        self.clip_id = clip_id
        self.specs: list[FrameSpec] = []

    def add(self, kind: str, **params: Any) -> None:
        cleaned: dict[str, Any] = {"clip_id": self.clip_id}
        for k, v in params.items():
            if isinstance(v, bool):
                cleaned[k] = v
            elif isinstance(v, float):
                cleaned[k] = _f(v)
            elif isinstance(v, (int, np.integer)):
                cleaned[k] = int(v)
            else:
                cleaned[k] = v
        self.specs.append(FrameSpec(index=len(self.specs), kind=kind, params=cleaned))

    def hold(self, n: int, kind: str, **params: Any) -> None:
        for _ in range(max(1, int(n))):
            self.add(kind, **params)

    def tween(self, n: int, kind: str, start: dict, end: dict, *, discrete: set[str] | None = None) -> None:
        disc = discrete or set()
        keys = set(start) | set(end)
        steps = max(2, int(n))
        for i in range(steps):
            u = i / (steps - 1)
            su = smoothstep(u)
            params: dict[str, Any] = {}
            for key in keys:
                va = start[key] if key in start else end[key]
                vb = end[key] if key in end else start[key]
                if key in disc or isinstance(va, str) or isinstance(vb, str):
                    params[key] = vb if su >= 0.999 else va
                elif isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
                    params[key] = lerp(float(va), float(vb), su)
                else:
                    params[key] = vb if su >= 0.5 else va
            self.add(kind, **params)


def _base(**kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = dict(
        data="sep",
        split=0.0,
        squash=0.0,
        x_scale=1.0,
        right="none",
        right_alpha=1.0,
        w1=float(OPT_W[0]),
        w2=float(OPT_W[1]),
        b=float(OPT_B),
        line_alpha=1.0,
        line_style="solid",
        strip=0.0,
        strip_mode="geom",
        prob_alpha=0.0,
        score_alpha=0.0,
        score_gain=1.0,
        focus=-1,
        focus_alpha=0.0,
        wrong_ring=0.0,
        inside_ring=0.0,
        sv_fade=0.0,
        sv_pulse=0.0,
        label_alpha=0.0,
        width_alpha=0.0,
        eq_alpha=0.0,
        point_alpha=1.0,
        perp_alpha=0.0,
        both_perp=0.0,
        perp_blend=1.0,
        scene_alpha=1.0,
        walk=0.0,
        alpha_scale=1.0,
        alpha_label=0.0,
        drive="",
        bound_t=1.0,
        sweep=0.0,
        carve=0.0,
        marker_alpha=0.0,
        ghost=0.0,
        b_slide=0.0,
        curve_alpha=0.0,
        smo_seg=0,
        smo_u=0.0,
        far_alpha=0.0,
        pair_u=0.0,
        slider_alpha=0.0,
        slider_u=0.0,
        c_value=0.02,
        slack_alpha=0.0,
        glow=0.0,
        glow_mode="hinge",
        hinge_alpha=0.0,
        logit_alpha=0.0,
        line2_alpha=0.0,
        spot=-1,
        spot_alpha=0.0,
        axis_alpha=1.0,
        xlabel_mode="hours",
        vec_alpha=1.0,
        cam_u=0.0,
        vox_grow=0.0,
        bisect_alpha=0.0,
        cons_n=2.0,
        extra=0.0,
        dual_kind="pair",
        quad_alpha=1.0,
        primal_alpha=0.0,
        formula_step=0.0,
        lag_step=0.0,
        dual_ghost=0.0,
        w1b=float(PAIR_W[0]),
        w2b=float(PAIR_W[1]),
        bb=float(PAIR_B),
        line2_color="#8a97a3",
        caption="",
        caption_alpha=0.0,
    )
    d.update(kw)
    return d


_DISC = {
    "data", "right", "line_style", "strip_mode", "glow_mode", "xlabel_mode",
    "drive", "focus", "spot", "dual_kind", "line2_color", "caption",
}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _dataset(pack, name: str):
    return pack[str(name)]


def _xy(pack, spec):
    ds = _dataset(pack, spec.get("data", "sep"))
    x = np.array(ds["x"], dtype=np.float64, copy=True)
    e = np.array(ds["e"], dtype=np.float64, copy=True)
    u = float(spec.get("pair_u", 0.0) or 0.0)
    if spec.get("data") == "real" and u > 1e-4:
        for i in (20, 21):
            x[i] = lerp(float(x[i]), 3.5, u)
            e[i] = lerp(float(e[i]), 3.5, u)
    scale = float(spec.get("x_scale", 1.0) or 1.0)
    return x * scale, e, np.asarray(ds["y01"]), np.asarray(ds["ypm"])


def _limits(spec):
    data = spec.get("data", "sep")
    if data == "toy":
        return (-3.8, 3.9), (-2.55, 2.55)
    scale = float(spec.get("x_scale", 1.0) or 1.0)
    return (0.0, 7.0 * scale), (0.0, 7.0)


def _vis_alpha(spec, n: int) -> np.ndarray:
    """Per-student visibility. Missing keys stay visible, except the two appended crossers."""
    if spec.get("data") in ("curve", "toy"):
        return np.ones(n, dtype=np.float64)
    out = np.ones(n, dtype=np.float64)
    for i in range(n):
        key = f"v{i}"
        if key in spec:
            out[i] = float(spec[key])
        elif i >= 20:
            out[i] = 0.0
    return out


def _point_alpha(spec, n, ypm, sv_index) -> np.ndarray:
    del ypm, sv_index
    base = float(spec.get("point_alpha", 1.0) or 0.0)
    return base * _vis_alpha(spec, n)


def _style_2d(ax, xlim, ylim, spec, *, font, axis_font, xlabel, ylabel) -> None:
    aa = float(np.clip(spec.get("axis_alpha", 1.0) or 0.0, 0.0, 1.0))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # One hour of study time matches one hour of exam length, so a right angle stays square.
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_xlabel(xlabel, fontsize=axis_font, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=axis_font, labelpad=8)
    ax.set_title("")
    ax.tick_params(labelsize=font * 0.92, width=1.1, length=5)
    ax.grid(True, alpha=0.18 * aa)
    for sp in ax.spines.values():
        sp.set_color("#444444")
        sp.set_linewidth(1.15)
        sp.set_alpha(aa)
    ax.xaxis.label.set_alpha(aa)
    ax.yaxis.label.set_alpha(aa)
    mode = spec.get("xlabel_mode", "hours")
    scale = float(spec.get("x_scale", 1.0) or 1.0)
    if (mode == "minutes" or scale > 1.8) and spec.get("data") != "toy":
        ticks = np.linspace(xlim[0], xlim[1], 6)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(round(t * 60.0 / max(scale, 1e-6)))) for t in ticks])
    elif mode != "x" and abs(xlim[1] - 7.0) < 0.2 and abs(ylim[1] - 7.0) < 0.2:
        ax.set_xticks(range(0, 8))
        ax.set_yticks(range(0, 8))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_alpha(aa)


def _labels_for(spec):
    mode = spec.get("xlabel_mode", "hours")
    scale = float(spec.get("x_scale", 1.0) or 1.0)
    if mode == "x" or spec.get("data") == "toy":
        return r"$x_1$", r"$x_2$"
    if mode == "minutes" or scale > 1.8:
        return "Study time (minutes)", "Exam length (hours)"
    return "Study time (hours)", "Exam length (hours)"


def _icons(ax, xs, ys, y01, alphas, check, cross, zoom) -> None:
    for i in range(len(xs)):
        if alphas[i] < 0.04:
            continue
        image = check if y01[i] > 0.5 else cross
        icon = OffsetImage(image, zoom=zoom * (0.92 + 0.08 * float(alphas[i])))
        icon.set_alpha(float(np.clip(alphas[i], 0.0, 1.0)))
        box = AnnotationBbox(icon, (float(xs[i]), float(ys[i])), frameon=False, zorder=6, pad=0.0)
        box.set_clip_on(False)
        ax.add_artist(box)


def _clip_line(w1, w2, b, xlim, ylim):
    x0, x1 = xlim
    y0, y1 = ylim
    pts: list[tuple[float, float]] = []

    def add(x, y):
        if x0 - 1e-5 <= x <= x1 + 1e-5 and y0 - 1e-5 <= y <= y1 + 1e-5:
            for p in pts:
                if abs(p[0] - x) < 1e-5 and abs(p[1] - y) < 1e-5:
                    return
            pts.append((float(x), float(y)))

    if abs(w2) > 1e-8:
        for x in (x0, x1):
            add(x, -(w1 * x + b) / w2)
    if abs(w1) > 1e-8:
        for y in (y0, y1):
            add(-(w2 * y + b) / w1, y)
    if len(pts) < 2:
        return None
    return pts[0], pts[1]


def _draw_line(ax, w1, w2, b, xlim, ylim, *, alpha, style, color, lw) -> None:
    if alpha < 0.03 or np.hypot(w1, w2) < 1e-6:
        return
    seg = _clip_line(w1, w2, b, xlim, ylim)
    if seg is None:
        return
    (xa, ya), (xb, yb) = seg
    ax.plot(
        [xa, xb], [ya, yb],
        color=color, lw=lw, ls="--" if style == "dashed" else "-",
        alpha=float(alpha), zorder=4, solid_capstyle="round",
    )


def _half_width(w1, w2, b, x, e, ypm, spec, vis=None) -> float:
    amount = float(spec.get("strip", 0.0) or 0.0)
    if amount <= 1e-4 or np.hypot(w1, w2) < 1e-6:
        return 0.0
    if spec.get("strip_mode") == "functional":
        return amount
    f = signed_scores(w1, w2, b, x, e, ypm)
    if vis is not None:
        f = np.where(np.asarray(vis) > 0.45, f, np.inf)
    pos = f[np.isfinite(f) & (f > 1e-5)]
    if pos.size == 0:
        return 0.0
    return float(pos.min()) * amount


def _draw_strip(ax, w1, w2, b, h, xlim, ylim, alpha) -> None:
    if alpha < 0.03 or h < 1e-5:
        return
    if abs(w2) >= abs(w1):
        xs = np.linspace(xlim[0], xlim[1], 220)
        y_pos = -(w1 * xs + b - h) / w2
        y_neg = -(w1 * xs + b + h) / w2
        ax.fill_between(xs, y_neg, y_pos, color=STRIP_COLOR, alpha=0.45 * alpha, zorder=0, linewidth=0)
        ax.plot(xs, y_pos, color=EDGE_COLOR, lw=1.5, ls="--", alpha=alpha, zorder=3)
        ax.plot(xs, y_neg, color=EDGE_COLOR, lw=1.5, ls="--", alpha=alpha, zorder=3)
    else:
        ys = np.linspace(ylim[0], ylim[1], 220)
        x_pos = -(w2 * ys + b - h) / w1
        x_neg = -(w2 * ys + b + h) / w1
        ax.fill_betweenx(ys, x_neg, x_pos, color=STRIP_COLOR, alpha=0.45 * alpha, zorder=0, linewidth=0)
        ax.plot(x_pos, ys, color=EDGE_COLOR, lw=1.5, ls="--", alpha=alpha, zorder=3)
        ax.plot(x_neg, ys, color=EDGE_COLOR, lw=1.5, ls="--", alpha=alpha, zorder=3)


def _foot(w1, w2, b, x, y):
    n2 = w1 * w1 + w2 * w2
    t = (w1 * x + w2 * y + b) / n2
    return np.array([x - t * w1, y - t * w2], dtype=np.float64)


def _vertical_foot(w1, w2, b, x, y):
    if abs(w2) < 1e-8:
        return _foot(w1, w2, b, x, y)
    return np.array([x, -(w1 * x + b) / w2], dtype=np.float64)


def _draw_seg(ax, p, q, *, color, lw, alpha, z=5) -> None:
    if alpha < 0.03:
        return
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, alpha=alpha, zorder=z, solid_capstyle="round")


def _ring(ax, x, y, *, color, alpha, size, lw) -> None:
    if alpha < 0.03:
        return
    ax.scatter(
        [x], [y], s=size, facecolors="none", edgecolors=color,
        linewidths=lw, alpha=alpha, zorder=7,
    )


def _prob_field(ax, pack, w1, w2, b, xlim, ylim, alpha) -> None:
    if alpha < 0.03:
        return
    n = int(pack.get("grid_n", 60))
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)
    scale = float((xlim[1] - xlim[0]) / 7.0) if xlim[1] > xlim[0] else 1.0
    # The field lives in hours. When the axis is stretched, study = x / scale.
    study = X / max(scale, 1e-6)
    Z = sigmoid(w1 * study + w2 * Y + b)
    ax.contourf(X, Y, Z, levels=18 if n < 50 else 26, cmap=CMAP, vmin=0.0, vmax=1.0, alpha=0.55 * alpha, zorder=0)


def _scores_text(ax, xs, ys, w1, w2, b, gain, alpha, *, scale, vis) -> None:
    if alpha < 0.04:
        return
    raw_x = np.asarray(xs) / max(scale, 1e-6)
    shown = (w1 * raw_x + w2 * np.asarray(ys) + b) * gain
    for i in range(len(xs)):
        if vis[i] < 0.45:
            continue
        ax.annotate(
            f"{shown[i]:+.1f}",
            (float(xs[i]), float(ys[i])),
            textcoords="offset points", xytext=(10, 8),
            fontsize=12, color="#222222", alpha=alpha, zorder=8,
        )


def _alpha_tags(ax, xs, ys, ypm, spec, vis) -> None:
    lab = float(spec.get("alpha_label", 0.0) or 0.0)
    if lab < 0.04:
        return
    for i in range(len(xs)):
        if vis[i] < 0.45:
            continue
        key = f"a{i}"
        if key not in spec:
            continue
        val = float(spec[key])
        if abs(val) < 0.015:
            continue
        dy = 22 if ypm[i] < 0 else -28
        ax.annotate(
            f"α={val:.2f}",
            (float(xs[i]), float(ys[i])),
            textcoords="offset points", xytext=(12, dy),
            fontsize=12, color="#333333", alpha=lab, zorder=8,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75 * lab),
        )


def _nearest_side(w1, w2, b, xs, ys, ypm, sign: float, vis) -> int | None:
    mask = (ypm * sign > 0) & (np.asarray(vis) > 0.45)
    if not np.any(mask):
        return None
    f = np.abs(w1 * np.asarray(xs) + w2 * np.asarray(ys) + b)
    f = np.where(mask, f, np.inf)
    return int(np.argmin(f))


def _draw_perps(ax, spec, w1, w2, b, xs, ys, ypm, h, xlim, ylim, vis) -> None:
    pa = float(spec.get("perp_alpha", 0.0) or 0.0)
    both = float(spec.get("both_perp", 0.0) or 0.0)
    wa = float(spec.get("width_alpha", 0.0) or 0.0)
    if pa < 0.03 and both < 0.03 and wa < 0.05:
        return
    blend = float(spec.get("perp_blend", 1.0) or 0.0)
    nrm = max(np.hypot(w1, w2), 1e-8)

    def seg_for(i: int):
        p = np.array([xs[i], ys[i]], dtype=np.float64)
        g = _foot(w1, w2, b, p[0], p[1])
        v = _vertical_foot(w1, w2, b, p[0], p[1])
        q = (1.0 - blend) * v + blend * g
        return p, q

    if both > 0.03:
        for sign in (1.0, -1.0):
            i = _nearest_side(w1, w2, b, xs, ys, ypm, sign, vis)
            if i is None:
                continue
            p, q = seg_for(i)
            _draw_seg(ax, p, q, color=NAVY, lw=2.4, alpha=both)
    elif pa > 0.03:
        focus = _idx(spec, "focus")
        hidden = focus < 0 or focus >= len(xs) or float(np.asarray(vis)[focus]) <= 0.45
        if hidden:
            f = np.abs(w1 * np.asarray(xs) + w2 * np.asarray(ys) + b)
            f = np.where(np.asarray(vis) > 0.45, f, np.inf)
            focus = int(np.argmin(f)) if np.isfinite(f).any() else -1
        if 0 <= focus < len(xs):
            p, q = seg_for(focus)
            _draw_seg(ax, p, q, color=NAVY, lw=2.6, alpha=pa)
        if 0 <= focus < len(xs) and float(spec.get("label_alpha", 0.0) or 0.0) > 0.05:
            direction = p - q
            length = float(np.linalg.norm(direction))
            if length < 1e-6:
                direction = np.array([0.35, 0.35])
            else:
                direction = direction / length * 0.42
            spot = p + direction
            ax.text(
                spot[0], spot[1], "margin", color=NAVY, fontsize=12,
                alpha=float(spec["label_alpha"]), zorder=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8),
            )

    if wa > 0.05 and h > 1e-4:
        seg = _clip_line(w1, w2, b, xlim, ylim)
        if seg is not None:
            a = np.array(seg[0])
            c = np.array(seg[1])
            p = 0.22 * a + 0.78 * c
            normal = np.array([w1, w2], dtype=np.float64) / nrm
            dist = h / nrm
            lo = p - dist * normal
            hi = p + dist * normal
            ax.annotate(
                "",
                xy=(hi[0], hi[1]),
                xytext=(lo[0], lo[1]),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.8, mutation_scale=12),
                zorder=8,
            )
            ax.annotate(
                "full width",
                (float(p[0]), float(p[1])),
                textcoords="offset points", xytext=(10, 12),
                color=NAVY, fontsize=12, alpha=wa, ha="left", va="bottom", zorder=8,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82),
            )


def _draw_walk(ax, spec, w1, w2, b, xlim, ylim) -> None:
    walk = float(spec.get("walk", 0.0) or 0.0)
    if walk < 0.02 or np.hypot(w1, w2) < 1e-6:
        return
    seg = _clip_line(w1, w2, b, xlim, ylim)
    if seg is None:
        return
    p = 0.5 * (np.array(seg[0]) + np.array(seg[1]))
    direction = np.array([w1, w2], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    q = p + walk * direction
    _draw_seg(ax, p, q, color="#c0392b", lw=2.6, alpha=1.0)
    ax.scatter([q[0]], [q[1]], s=36, c="#c0392b", zorder=8)
    ax.annotate(
        rf"$+\|w\|$",
        (q[0], q[1]),
        textcoords="offset points", xytext=(8, 8),
        fontsize=13, color="#c0392b", zorder=8,
    )


def _draw_bisect(ax, alpha: float) -> None:
    """Segment joining the two kept students, and a right angle where the bisector crosses it."""
    if alpha < 0.04:
        return
    ax.plot([2.0, 4.0], [4.0, 2.0], color=NAVY, lw=1.7, ls=":", alpha=alpha, zorder=5)
    center = np.array([3.0, 3.0])
    along = np.array([1.0, -1.0], dtype=np.float64)
    along = along / np.linalg.norm(along) * 0.32
    across = np.array([1.0, 1.0], dtype=np.float64)
    across = across / np.linalg.norm(across) * 0.32
    p = center + along
    q = center + along + across
    r = center + across
    ax.plot([p[0], q[0], r[0]], [p[1], q[1], r[1]], color=NAVY, lw=1.5, alpha=alpha, zorder=5)
    ax.scatter([center[0]], [center[1]], s=22, c=NAVY, alpha=alpha, zorder=5)


def _draw_students(ax, pack, spec, *, zoom, font, axis_font) -> None:
    xs, ys, y01, ypm = _xy(pack, spec)
    xlim, ylim = _limits(spec)
    w1 = float(spec.get("w1", 1.0))
    w2 = float(spec.get("w2", -1.0))
    b = float(spec.get("b", 0.0))
    if spec.get("drive") == "alpha":
        t = float(spec.get("alpha_scale", 1.0) or 0.0)
        w1, w2, b = t, -t, 0.0
    b = b + float(spec.get("b_slide", 0.0) or 0.0)
    xlabel, ylabel = _labels_for(spec)
    _style_2d(ax, xlim, ylim, spec, font=font, axis_font=axis_font, xlabel=xlabel, ylabel=ylabel)
    _prob_field(ax, pack, w1, w2, b, xlim, ylim, float(spec.get("prob_alpha", 0.0) or 0.0))
    # Weights are in hours. Plotted x may be stretched, so the drawn line uses w1/scale.
    scale = float(spec.get("x_scale", 1.0) or 1.0)
    raw_x = xs / max(scale, 1e-6)
    w1_draw = w1 / max(scale, 1e-6)
    vis = _vis_alpha(spec, len(xs))
    h = _half_width(w1, w2, b, raw_x, ys, ypm, spec, vis)
    edge_a = float(spec.get("line_alpha", 1.0) or 0.0)
    _draw_strip(ax, w1_draw, w2, b, h, xlim, ylim, edge_a if h > 0 else 0.0)
    style = spec.get("line_style", "solid")
    if style == "auto":
        style = "dashed" if float(spec.get("prob_alpha", 0.0) or 0.0) > 0.35 else "solid"
    _draw_line(ax, w1_draw, w2, b, xlim, ylim, alpha=edge_a, style=style, color=LINE_COLOR, lw=2.7)
    if float(spec.get("line2_alpha", 0.0) or 0.0) > 0.03:
        _draw_line(
            ax,
            float(spec.get("w1b", 0.7)),
            float(spec.get("w2b", -0.7)),
            float(spec.get("bb", 0.0)),
            xlim, ylim,
            alpha=float(spec["line2_alpha"]),
            style="dashed",
            color=str(spec.get("line2_color", LOGIT_COLOR)),
            lw=2.4,
        )
    _draw_perps(ax, spec, w1_draw, w2, b, xs, ys, ypm, h, xlim, ylim, vis)
    _draw_walk(ax, spec, w1_draw, w2, b, xlim, ylim)
    _draw_bisect(ax, float(spec.get("bisect_alpha", 0.0) or 0.0))

    alphas = _point_alpha(spec, len(xs), ypm, None)
    glow = float(spec.get("glow", 0.0) or 0.0)
    if glow > 0.03:
        f = signed_scores(w1, w2, b, raw_x, ys, ypm)
        if spec.get("glow_mode") == "logistic":
            loss = logistic_loss(f)
            color = LOGIT_COLOR
        else:
            loss = np.maximum(0.0, 1.0 - f)
            color = NAVY
        sizes = 80.0 + 520.0 * np.clip(loss / 1.6, 0.0, 1.0)
        ax.scatter(xs, ys, s=sizes, c=color, alpha=0.28 * glow, linewidths=0, zorder=2)

    _icons(ax, xs, ys, y01, alphas, pack["check"], pack["cross"], zoom)

    f_signed = signed_scores(w1, w2, b, raw_x, ys, ypm)
    wr = float(spec.get("wrong_ring", 0.0) or 0.0)
    ir = float(spec.get("inside_ring", 0.0) or 0.0)
    if wr > 0.03 or ir > 0.03:
        for i in range(len(xs)):
            if alphas[i] < 0.2:
                continue
            col = PASS_COLOR if ypm[i] > 0 else FAIL_COLOR
            if f_signed[i] <= 0.02 and wr > 0.03:
                _ring(ax, xs[i], ys[i], color=col, alpha=wr, size=430, lw=2.2)
            elif 0.02 < f_signed[i] < 0.98 and ir > 0.03:
                _ring(ax, xs[i], ys[i], color=NAVY, alpha=ir * 0.9, size=360, lw=1.6)

    fa = float(spec.get("focus_alpha", 0.0) or 0.0)
    focus = _idx(spec, "focus")
    if fa > 0.03 and 0 <= focus < len(xs) and vis[focus] > 0.4:
        col = PASS_COLOR if y01[focus] > 0.5 else FAIL_COLOR
        _ring(ax, xs[focus], ys[focus], color=col, alpha=fa, size=980, lw=2.8)

    pulse = float(spec.get("sv_pulse", 0.0) or 0.0)
    if pulse > 0.03:
        for i in (CORE_FAIL, CORE_PASS, MID_PT):
            if i >= len(xs) or vis[i] < 0.45:
                continue
            if float(spec.get(f"a{i}", 0.0) or 0.0) <= 0.05:
                continue
            col = PASS_COLOR if y01[i] > 0.5 else FAIL_COLOR
            _ring(ax, xs[i], ys[i], color=col, alpha=pulse, size=520, lw=2.4)

    spot = _idx(spec, "spot")
    sa = float(spec.get("spot_alpha", 0.0) or 0.0)
    if sa > 0.03 and 0 <= spot < len(xs):
        _ring(ax, xs[spot], ys[spot], color=GOLD, alpha=sa, size=640, lw=2.6)
        xi = max(0.0, 1.0 - float(f_signed[spot]))
        ax.annotate(
            rf"$\xi={xi:.1f}$",
            (float(xs[spot]), float(ys[spot])),
            textcoords="offset points", xytext=(14, 14),
            fontsize=14, color="#222222", alpha=sa, zorder=9,
        )

    _scores_text(
        ax, xs, ys, w1, w2, b,
        float(spec.get("score_gain", 1.0) or 1.0),
        float(spec.get("score_alpha", 0.0) or 0.0),
        scale=scale,
        vis=vis,
    )
    _alpha_tags(ax, xs, ys, ypm, spec, vis)

    cap_a = float(spec.get("caption_alpha", 0.0) or 0.0)
    cap = str(spec.get("caption", "") or "")
    if cap and cap_a > 0.04:
        ax.text(
            0.98, 0.04, cap,
            transform=ax.transAxes, ha="right", va="bottom", fontsize=13,
            color="#222222", alpha=cap_a, zorder=9,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88 * cap_a),
        )
    eq = float(spec.get("eq_alpha", 0.0) or 0.0)
    if eq > 0.04:
        ax.text(
            0.98, 0.97,
            rf"$w_1={w1:.2f}$" + "\n" + rf"$w_2={w2:.2f}$" + "\n" + rf"$b={b:.2f}$",
            transform=ax.transAxes, va="top", ha="right", fontsize=14, color="#222222", alpha=eq, zorder=9,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88 * eq),
        )
    # The half-width is named on the perpendicular itself, not in a corner.

    if float(spec.get("slider_alpha", 0.0) or 0.0) > 0.04:
        _slider(ax, spec, raw_x, ys, ypm, w1, w2, b, vis)


def _slider(ax, spec, raw_x, ys, ypm, w1, w2, b, vis) -> None:
    a = float(spec.get("slider_alpha", 0.0) or 0.0)
    u = float(np.clip(spec.get("slider_u", 0.0) or 0.0, 0.0, 1.0))
    x0, x1 = 0.8, 6.2
    y = 0.32
    ax.plot([x0, x1], [y, y], color="white", lw=7.0, alpha=0.9 * a, zorder=7, solid_capstyle="round")
    ax.plot([x0, x1], [y, y], color="#1a1a1a", lw=2.6, alpha=a, zorder=8, solid_capstyle="round")
    x = lerp(x0, x1, u)
    ax.scatter([x], [y], s=70, c=NAVY, zorder=9, alpha=a)
    cval = float(spec.get("c_value", 0.2) or 0.2)
    ax.text(x, y + 0.28, rf"$C={cval:.2f}$", ha="center", va="bottom", fontsize=12, color=NAVY, alpha=a, zorder=9)
    if float(spec.get("slack_alpha", 0.0) or 0.0) > 0.04:
        f = signed_scores(w1, w2, b, raw_x, ys, ypm)
        total = float(np.sum(np.maximum(0.0, 1.0 - f)[np.asarray(vis) > 0.45]))
        ax.text(
            0.98, 0.97, rf"$\sum\xi={total:.1f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=14, color="#222222",
            alpha=float(spec["slack_alpha"]),
        )


def _square_rect(rect: list[float]) -> list[float]:
    """Largest square, in inches, centered in a figure-fraction rectangle.

    The frame stays 15 by 9.5. Equal data limits then share one screen scale.
    """
    x, y, w, h = rect
    side = min(w * FIGSIZE[0], h * FIGSIZE[1])
    w2 = side / FIGSIZE[0]
    h2 = side / FIGSIZE[1]
    return [x + 0.5 * (w - w2), y + 0.5 * (h - h2), w2, h2]


def _rect_left(split: float, squash: float) -> list[float]:
    s = smoothstep(split)
    q = smoothstep(squash)
    x = lerp(SCATTER[0], DUO_L[0], s)
    y = lerp(SCATTER[1], DUO_L[1], s)
    w = lerp(SCATTER[2], DUO_L[2], s)
    h = lerp(SCATTER[3], DUO_L[3], s)
    h2 = lerp(h, h / 4.4, q)
    y2 = y + 0.5 * (h - h2)
    return _square_rect([x, y2, w, h2])


def _rect_right(split: float) -> list[float]:
    s = smoothstep(split)
    return [lerp(1.08, DUO_R[0], s), DUO_R[1], DUO_R[2], DUO_R[3]]


def _layout_rects(split: float, squash: float, grow: float) -> tuple[list[float], list[float]]:
    """Student square on the left. ``grow`` enlarges the right-hand 3D landscape."""
    g = smoothstep(grow)
    left0 = _rect_left(split, squash)
    right0 = _rect_right(split)
    left1 = _square_rect([0.028, 0.19, 0.42, 0.64])
    right1 = [0.45, 0.015, 0.54, 0.97]
    left = [lerp(a, b, g) for a, b in zip(left0, left1)]
    right = [lerp(a, b, g) for a, b in zip(right0, right1)]
    return left, right


def _draw_vectors(ax, pack, spec, font: float) -> None:
    """Head-to-tail sum α y x for the students whose coefficients are on screen."""
    alpha = float(spec.get("right_alpha", 1.0) or 0.0)
    if alpha < 0.03:
        return
    ds = pack["sep"]
    order = (CORE_FAIL, CORE_PASS, MID_PT, LEFT_PT, RIGHT_PT)
    arrows = []
    for i in order:
        key = f"a{i}"
        if key not in spec:
            continue
        coef = float(spec[key])
        if abs(coef) < 0.02:
            continue
        vec = coef * float(ds["ypm"][i]) * np.array([ds["x"][i], ds["e"][i]], dtype=np.float64)
        color = FAIL_COLOR if ds["ypm"][i] < 0 else PASS_COLOR
        arrows.append((vec, color))
    ax.set_xlim(-1.5, 1.8)
    ax.set_ylim(-2.3, 1.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$w_1$", fontsize=font)
    ax.set_ylabel(r"$w_2$", fontsize=font)
    ax.set_title("")
    ax.axhline(0, color="#cccccc", lw=1.0, zorder=0)
    ax.axvline(0, color="#cccccc", lw=1.0, zorder=0)
    cursor = np.zeros(2)
    for vec, color in arrows:
        tip = cursor + vec
        ax.annotate(
            "",
            xy=tip,
            xytext=cursor,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4, alpha=alpha),
        )
        cursor = tip
    if arrows:
        ax.scatter([cursor[0]], [cursor[1]], s=70, c=LINE_COLOR, zorder=5, alpha=alpha)
        ax.text(cursor[0] + 0.08, cursor[1] + 0.12, r"$w$", color="#222222", fontsize=14, alpha=alpha)
    step = float(spec.get("lag_step", 0.0) or 0.0)
    lines = []
    if step > 0.15:
        lines.append(r"$\frac{1}{2}\|w\|^2$")
    if step > 0.7:
        lines.append(r"$+\alpha_1[1-y_1 f(x_1)]$")
    if step > 1.3:
        lines.append(r"$+\alpha_2[1-y_2 f(x_2)]$")
    if step > 2.0:
        lines.append(r"$w=\sum \alpha\, y\, x$")
    if step > 2.6:
        lines.append(r"$\alpha_1=\alpha_2$")
    if lines:
        ax.text(
            0.04, 0.97, "\n".join(lines),
            transform=ax.transAxes, va="top", ha="left", fontsize=13, color="#222222", alpha=alpha,
        )
    for sp in ax.spines.values():
        sp.set_alpha(alpha)
    ax.tick_params(labelsize=font * 0.8)


def _draw_slice(ax, pack, spec, font, axis_font) -> None:
    """(w1, w2) slice. Contours are the objective. Red wash is a forbidden half-plane."""
    a = float(spec.get("right_alpha", 1.0) or 0.0)
    if a < 0.03:
        return
    ds = pack["sep"]
    b = float(spec.get("b", 0.0) or 0.0)
    rows = [CORE_FAIL, CORE_PASS]
    if float(spec.get("extra", 0.0) or 0.0) > 0.02 or float(spec.get("cons_n", 2.0) or 0.0) > 2.02:
        rows.append(MID_PT)
    cons_n = float(spec.get("cons_n", float(len(rows))) or 0.0)
    w1g = np.linspace(-0.15, 1.55, 200)
    w2g = np.linspace(-1.70, 0.30, 200)
    W1, W2 = np.meshgrid(w1g, w2g)
    obj = 0.5 * (W1 ** 2 + W2 ** 2)
    ax.contour(
        W1, W2, obj,
        levels=[0.08, 0.2, 0.35, 0.55, 0.8, 1.15],
        colors="#1f4e79", linewidths=1.05, alpha=0.55 * a, zorder=1,
    )
    feasible = np.ones(W1.shape, dtype=bool)
    for i, idx in enumerate(rows):
        wa = float(np.clip(cons_n - i, 0.0, 1.0))
        if wa < 0.03:
            continue
        f = float(ds["ypm"][idx]) * (
            W1 * float(ds["x"][idx]) + W2 * float(ds["e"][idx]) + b
        )
        ax.contourf(W1, W2, f, levels=[-30.0, 1.0], colors=["#f6d0d0"], alpha=0.38 * wa * a, zorder=0)
        ax.contour(W1, W2, f, levels=[1.0], colors=["#1f4e79"], linewidths=1.7, alpha=wa * a, zorder=2)
        if wa > 0.8:
            feasible &= f >= 0.98
    w1 = float(spec.get("w1", 0.5))
    w2 = float(spec.get("w2", -0.5))
    # Recompute the marker's own constraints so a half-revealed cut doesn't flicker the color.
    yf = np.inf
    for i, idx in enumerate(rows):
        if float(np.clip(cons_n - i, 0.0, 1.0)) < 0.8:
            continue
        yf = min(yf, float(ds["ypm"][idx]) * (w1 * float(ds["x"][idx]) + w2 * float(ds["e"][idx]) + b))
    color = GOLD if yf >= 0.98 else FAIL_COLOR
    ax.scatter([w1], [w2], s=90, c=color, edgecolors="white", linewidths=1.2, zorder=5, alpha=a)
    ax.text(
        0.04, 0.96, rf"$b={b:.2f}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=13, color="#222222", alpha=a,
    )
    ax.set_xlim(-0.15, 1.55)
    ax.set_ylim(-1.70, 0.30)
    ax.set_xlabel(r"$w_1$", fontsize=axis_font * 0.9)
    ax.set_ylabel(r"$w_2$", fontsize=axis_font * 0.9)
    ax.set_title("")
    ax.tick_params(labelsize=font * 0.8)
    for sp in ax.spines.values():
        sp.set_alpha(a)


def _alpha_geometry(kind: str):
    """Shared-α picture. Bowl is weight size. Wall is where the strip just touches."""
    if kind == "moved":
        wall, bowl_coef = 0.40, 2.5
        def bowl(t):
            tt = np.asarray(t, dtype=np.float64)
            return bowl_coef * tt * tt
        def agree(t):
            tt = np.asarray(t, dtype=np.float64)
            return 2.0 * tt - bowl_coef * tt * tt
    else:
        wall, bowl_coef = 0.25, 4.0
        def bowl(t):
            tt = np.asarray(t, dtype=np.float64)
            return bowl_coef * tt * tt
        def agree(t):
            tt = np.asarray(t, dtype=np.float64)
            return 2.0 * tt - bowl_coef * tt * tt
    return wall, bowl, agree


def _draw_dual(ax, spec, font, axis_font) -> None:
    """One slider, shared α.

    The rising curve is how long the weights are. Left of the wall the strip
    covers the edge students. The best α is the bottom of the allowed side.
    A second curve comes up from below and meets that point.
    """
    a = float(spec.get("right_alpha", 1.0) or 0.0)
    if a < 0.03:
        return
    kind = spec.get("dual_kind", "pair")
    ghost = float(spec.get("dual_ghost", 0.0) or 0.0)
    step = float(spec.get("formula_step", 0.0) or 0.0)
    wall, bowl, agree = _alpha_geometry(kind)
    ts = np.linspace(0.0, 0.58, 180)
    reveal = float(np.clip(spec.get("quad_alpha", 1.0) or 0.0, 0.0, 1.0))

    if ghost > 0.04:
        other = "pair" if kind == "moved" else "moved"
        owall, obowl, _oagree = _alpha_geometry(other)
        ax.plot(ts, obowl(ts), color="#9aa7b5", lw=1.6, ls="--", alpha=ghost * a, zorder=1)
        ax.axvline(owall, color="#9aa7b5", lw=1.2, ls="--", alpha=ghost * a, zorder=1)
        ax.text(
            owall, 1.02, "before",
            color="#8a97a3", fontsize=12, ha="center", va="bottom", alpha=ghost * a, zorder=4,
        )
    bowl_u = float(np.clip((step - 0.15) / 0.7, 0.0, 1.0)) * reveal
    if bowl_u > 0.02:
        n = max(2, int(bowl_u * len(ts)))
        ax.plot(ts[:n], bowl(ts[:n]), color=NAVY, lw=2.5, alpha=a, zorder=3)
    pink_u = float(np.clip((step - 1.05) / 0.6, 0.0, 1.0))
    if pink_u > 0.02:
        ax.axvspan(0.0, wall, color="#f6d0d0", alpha=0.55 * a * pink_u, zorder=0)
        ax.axvline(wall, color=NAVY, lw=1.6, alpha=a * pink_u, zorder=2)
        ax.text(
            max(wall * 0.42, 0.08), 0.92, "not allowed",
            color="#a33b3b", fontsize=12, ha="center", va="top", alpha=a * pink_u, zorder=4,
        )
    line_a = float(spec.get("primal_alpha", 0.0) or 0.0) * a
    curve_u = float(np.clip((step - 2.05) / 0.6, 0.0, 1.0)) * line_a
    if curve_u > 0.03:
        ax.plot(ts, agree(ts), color=GOLD, lw=2.3, alpha=curve_u, zorder=3)
    t = float(spec.get("bound_t", 0.0) or 0.0)
    if bowl_u > 0.05:
        on_bowl = float(bowl(t))
        allowed = t + 1e-6 >= wall
        color = GOLD if allowed else FAIL_COLOR
        ax.scatter([t], [on_bowl], s=110, c=color, edgecolors="white", linewidths=1.4, zorder=6, alpha=a)
        if curve_u > 0.45 and abs(t - wall) < 0.04:
            ax.annotate(
                "best",
                (wall, float(bowl(wall))),
                textcoords="offset points", xytext=(10, 12),
                fontsize=13, color="#222222", alpha=curve_u, zorder=7,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9),
            )
    ax.set_xlim(-0.02, 0.62)
    ax.set_ylim(-0.04, 1.15)
    ax.set_xlabel(r"shared $\alpha$", fontsize=axis_font * 0.85)
    ax.set_ylabel("weight size", fontsize=axis_font * 0.85)
    ax.set_title("")
    ax.tick_params(labelsize=font * 0.85)
    for sp in ax.spines.values():
        sp.set_alpha(a)


def _draw_loss(ax, pack, spec, font, axis_font) -> None:
    a = float(spec.get("right_alpha", 1.0) or 0.0)
    if a < 0.03:
        return
    m = np.linspace(-1.6, 4.2, int(pack.get("loss_n", 120)))
    ha = float(spec.get("hinge_alpha", 0.0) or 0.0) * a
    la = float(spec.get("logit_alpha", 0.0) or 0.0) * a
    if ha > 0.03:
        y = np.maximum(0.0, 1.0 - m)
        ax.plot(m, y, color=HINGE_COLOR, lw=2.8, alpha=ha)
        ax.text(1.35, 0.22, "hinge", color=HINGE_COLOR, fontsize=13, alpha=ha, va="bottom")
    if la > 0.03:
        y = logistic_loss(m)
        ax.plot(m, y, color=LOGIT_COLOR, lw=2.8, alpha=la)
        ax.text(2.15, float(logistic_loss(np.array([2.15]))[0]) + 0.18, "logistic", color=LOGIT_COLOR, fontsize=13, alpha=la)
    ax.axvline(0.0, color="#bbbbbb", lw=1.0, zorder=0)
    ax.axvline(1.0, color="#bbbbbb", lw=1.0, ls="--", zorder=0)
    ax.set_xlim(-1.7, 5.1)
    ax.set_ylim(-0.08, 2.9)
    ax.set_xlabel("signed score", fontsize=axis_font * 0.85)
    ax.set_ylabel("loss", fontsize=axis_font * 0.85)
    ax.set_title("")
    ax.tick_params(labelsize=font * 0.85)


def _draw_parabola(fig, rect, spec, font) -> None:
    a = float(spec.get("curve_alpha", 0.0) or 0.0)
    if a < 0.04:
        return
    ix = rect[0] + rect[2] * 0.56
    iy = rect[1] + rect[3] * 0.52
    iw = rect[2] * 0.40
    ih = rect[3] * 0.40
    ax = fig.add_axes([ix, iy, iw, ih])
    seg = int(spec.get("smo_seg", 0) or 0)
    axis, vals, _peak = smo_curve(seg, n=60)
    ax.plot(axis, vals, color=NAVY, lw=2.0, alpha=a)
    u = float(spec.get("smo_u", 0.0) or 0.0)
    # Dot at the current step's coordinate.
    if seg <= 0:
        x = 0.4 * u
    elif seg == 1:
        x = 0.16 * u
    elif seg == 2:
        x = 0.12 * u
    else:
        x = u
    y = float(np.interp(x, axis, vals))
    ax.scatter([x], [y], s=36, c=GOLD, zorder=4, alpha=a)
    ax.set_title("")
    ax.set_xlabel("pair step", fontsize=font * 0.7)
    ax.set_ylabel("dual", fontsize=font * 0.7)
    ax.tick_params(labelsize=font * 0.62)
    for sp in ax.spines.values():
        sp.set_alpha(a)


def _draw_voxels(ax, pack, spec) -> None:
    a = float(spec.get("right_alpha", 1.0) or 0.0)
    if a < 0.03:
        return
    v = pack["voxel"]
    sweep = float(np.clip(spec.get("sweep", 0.0) or 0.0, 0.0, 1.0))
    sweep = smoothstep(sweep)
    carve = float(np.clip(spec.get("carve", 0.0) or 0.0, 0.0, 1.0))
    show = v["proj"] <= (sweep * float(v["max_proj"]) + 1e-8)
    base = np.array(v["alpha0"], dtype=np.float64, copy=True)
    extra = float(np.clip(spec.get("extra", 0.0) or 0.0, 0.0, 1.0))
    yf = np.asarray(v["min_yf"], dtype=np.float64)
    if extra > 1e-4 and "min_yf_tight" in v:
        yf = (1.0 - extra) * yf + extra * np.asarray(v["min_yf_tight"], dtype=np.float64)
    violated = yf < 0.98
    # The whole trough stays visible. Infeasible cells fade to a ghost;
    # the allowed set stays solid so the constraint reads as a cut.
    viol_alpha = base * a * (0.10 + 0.90 * (1.0 - carve))
    feas_alpha = base * a * (1.0 + 0.2 * carve)
    alpha = np.where(violated, viol_alpha, feas_alpha)
    w1 = float(spec.get("w1", 1.0))
    w2 = float(spec.get("w2", -1.0))
    b = float(spec.get("b", 0.0)) + float(spec.get("b_slide", 0.0) or 0.0)
    if spec.get("drive") == "alpha":
        t = float(spec.get("alpha_scale", 1.0) or 0.0)
        w1, w2, b = t, -t, 0.0
    ma = float(spec.get("marker_alpha", 0.0) or 0.0)
    dx = float(v["dx"])
    dy = float(v["dy"])
    dz = float(v["dz"])
    # Drop the cubes around the marker so it sits on the landscape.
    if ma > 0.15:
        reach = 2.6 * max(dx, dy, dz)
        near = (
            (np.abs(v["x"] + 0.5 * dx - w1) < reach)
            & (np.abs(v["y"] + 0.5 * dy - w2) < reach)
            & (np.abs(v["z"] + 0.5 * dz - b) < reach)
        )
        alpha = np.array(alpha, copy=True)
        alpha[near] = 0.0
        extra_u = float(spec.get("extra", 0.0) or 0.0)
        if extra_u > 0.5 and abs(w1 - float(PAIR_W[0])) + abs(w2 - float(PAIR_W[1])) > 0.15:
            reach_old = 1.8 * max(dx, dy, dz)
            old = (
                (np.abs(v["x"] + 0.5 * dx - float(PAIR_W[0])) < reach_old)
                & (np.abs(v["y"] + 0.5 * dy - float(PAIR_W[1])) < reach_old)
                & (np.abs(v["z"] + 0.5 * dz - float(PAIR_B)) < reach_old)
            )
            alpha[old] *= 0.12
    show = show & (alpha > 0.025)
    ghost_m = show & violated
    solid_m = show & ~violated

    def _bars(mask):
        if not np.any(mask):
            return
        cols = np.column_stack([v["rgb"][mask], alpha[mask]])
        ax.bar3d(
            v["x"][mask], v["y"][mask], v["z"][mask],
            dx, dy, dz,
            color=cols, shade=False, linewidth=0.0, edgecolor=(0, 0, 0, 0),
        )

    _bars(ghost_m)
    _bars(solid_m)
    lim = v["lim"]
    ax.set_xlim(float(lim[0]), float(lim[1]))
    ax.set_ylim(float(lim[2]), float(lim[3]))
    ax.set_zlim(float(lim[4]), float(lim[5]))
    cam = float(spec.get("cam_u", 0.0) or 0.0)
    ax.view_init(elev=lerp(16.0, 22.0, cam), azim=lerp(-72.0, -48.0, cam))
    ax.set_xlabel(r"$w_1$", fontsize=11, labelpad=4)
    ax.set_ylabel(r"$w_2$", fontsize=11, labelpad=4)
    ax.set_zlabel(r"$b$", fontsize=11, labelpad=2)
    ax.set_title("")
    ax.tick_params(labelsize=8, pad=1)
    try:
        ax.set_box_aspect((lim[1] - lim[0], lim[3] - lim[2], lim[5] - lim[4]), zoom=1.08)
    except TypeError:
        try:
            ax.set_box_aspect((lim[1] - lim[0], lim[3] - lim[2], lim[5] - lim[4]))
        except Exception:
            pass
    except Exception:
        pass
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor("#e6e6e6")
        except Exception:
            pass
    if ma > 0.03:
        # Gold on the allowed side, red if this setting breaks a student.
        ds = pack["sep"]
        use = pack["tight_idx"] if float(spec.get("extra", 0.0) or 0.0) >= 0.5 else pack["core_idx"]
        yf = float(np.min(signed_scores(w1, w2, b, ds["x"][use], ds["e"][use], ds["ypm"][use])))
        color = GOLD if yf >= 0.98 else FAIL_COLOR
        d = 0.20
        ax.plot([w1 - d, w1 + d], [w2, w2], [b, b], color=color, lw=2.8, alpha=ma, zorder=12)
        ax.plot([w1, w1], [w2 - d, w2 + d], [b, b], color=color, lw=2.8, alpha=ma, zorder=12)
        ax.plot([w1, w1], [w2, w2], [b - d, b + d], color=color, lw=2.8, alpha=ma, zorder=12)
        ax.scatter(
            [w1], [w2], [b], s=1600, c="white", depthshade=False,
            edgecolors="none", alpha=0.95 * ma, zorder=13,
        )
        ax.scatter(
            [w1], [w2], [b], s=520, c=color, depthshade=False,
            edgecolors="white", linewidths=2.0, alpha=ma, zorder=14,
        )
        extra_u = float(spec.get("extra", 0.0) or 0.0)
        if extra_u > 0.5 and abs(w1 - float(PAIR_W[0])) + abs(w2 - float(PAIR_W[1])) > 0.15:
            ax.scatter(
                [float(PAIR_W[0])], [float(PAIR_W[1])], [float(PAIR_B)],
                s=160, c=FAIL_COLOR, depthshade=False,
                edgecolors="white", linewidths=0.8, alpha=0.55 * ma, zorder=11,
            )
    ax.text2D(
        0.02, 0.02, r"color $= \frac{1}{2}\|w\|^2$",
        transform=ax.transAxes, fontsize=10, color="#333333", alpha=min(1.0, a),
    )
    ghost = float(spec.get("ghost", 0.0) or 0.0)
    if ghost > 0.03:
        ax.plot([0.5, 0.16], [-0.5, -0.16], [0.0, 0.0], color=FAIL_COLOR, lw=2.0, alpha=ghost)
        ax.scatter([0.16], [-0.16], [0.0], s=50, c=FAIL_COLOR, depthshade=False, alpha=ghost, zorder=20)


def _apply_weights(spec) -> dict[str, Any]:
    """Copy spec and bake drive / b-slide into w1, w2, b for the student panel."""
    out = dict(spec)
    if out.get("drive") == "alpha":
        t = float(out.get("alpha_scale", 1.0) or 0.0)
        out["w1"], out["w2"], out["b"] = t, -t, 0.0
    else:
        out["b"] = float(out.get("b", 0.0) or 0.0) + float(out.get("b_slide", 0.0) or 0.0)
    out["b_slide"] = 0.0
    out["drive"] = ""
    return out


def _fade_image(img, alpha: float):
    if alpha >= 0.999:
        return img
    arr = np.asarray(img).astype(np.float32)
    arr = arr * float(alpha) + 255.0 * (1.0 - float(alpha))
    from PIL import Image

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _render_stage(pack, spec):
    import matplotlib.pyplot as plt

    from hita.export.raster import fig_to_image

    fig = plt.figure(figsize=pack.get("figsize", FIGSIZE), facecolor="white")
    split = float(spec.get("split", 0.0) or 0.0)
    squash = float(spec.get("squash", 0.0) or 0.0)
    grow = float(spec.get("vox_grow", 0.0) or 0.0)
    left, right_rect = _layout_rects(split, squash, grow)
    ax = fig.add_axes(left)
    font = float(pack.get("font_size", FONT))
    axis_font = float(pack.get("axis_label_size", AXIS_FONT))
    zoom = FULL_ZOOM * (0.86 if split > 0.55 else 1.0)
    zoom *= lerp(1.0, 0.42, smoothstep(squash))
    zoom *= lerp(1.0, 0.84, smoothstep(grow))
    baked = _apply_weights(spec)
    _draw_students(ax, pack, baked, zoom=zoom, font=font, axis_font=axis_font)
    if spec.get("data") == "toy":
        _draw_parabola(fig, left, spec, font)
    right_name = spec.get("right", "none")
    ra = float(spec.get("right_alpha", 1.0) or 0.0)
    if right_name and right_name != "none" and split > 0.04 and ra > 0.02:
        rect = right_rect
        if right_name == "voxel":
            axr = fig.add_axes(rect, projection="3d")
            _draw_voxels(axr, pack, baked)
        elif right_name == "vectors":
            axr = fig.add_axes(rect)
            _draw_vectors(axr, pack, baked, font)
        elif right_name == "dual":
            axr = fig.add_axes(rect)
            _draw_dual(axr, baked, font, axis_font)
        elif right_name == "slice":
            axr = fig.add_axes(rect)
            _draw_slice(axr, pack, baked, font, axis_font)
        elif right_name == "loss":
            axr = fig.add_axes(rect)
            _draw_loss(axr, pack, spec, font, axis_font)
    img = fig_to_image(fig, dpi=int(pack.get("dpi", 120)))
    return _fade_image(img, float(spec.get("scene_alpha", 1.0) or 1.0))


def _trio_rects(focus: float):
    u = smoothstep(focus)
    gap = 0.03
    left = 0.045
    width = 0.29
    base = []
    for i in range(3):
        base.append([left + i * (width + gap), 0.15, width, 0.72])
    rects = []
    alphas = []
    for i, r in enumerate(base):
        if i == 2:
            rects.append([
                lerp(r[0], SCATTER[0], u),
                lerp(r[1], SCATTER[1], u),
                lerp(r[2], SCATTER[2], u),
                lerp(r[3], SCATTER[3], u),
            ])
            alphas.append(1.0)
        else:
            rects.append([
                lerp(r[0], r[0] - 0.15, u),
                r[1],
                r[2],
                r[3],
            ])
            alphas.append(1.0 - u)
    return rects, alphas


def _render_trio(pack, spec):
    import matplotlib.pyplot as plt

    from hita.export.raster import fig_to_image

    fig = plt.figure(figsize=pack.get("figsize", FIGSIZE), facecolor="white")
    focus = float(spec.get("focus_panel", 0.0) or 0.0)
    rects, alphas = _trio_rects(focus)
    lines = (
        (float(spec.get("w1_0", _NARROW[0])), float(spec.get("w2_0", _NARROW[1])), float(spec.get("b_0", _NARROW[2]))),
        (float(spec.get("w1_1", _MEDIUM[0])), float(spec.get("w2_1", _MEDIUM[1])), float(spec.get("b_1", _MEDIUM[2]))),
        (float(spec.get("w1_2", _WIDE[0])), float(spec.get("w2_2", _WIDE[1])), float(spec.get("b_2", _WIDE[2]))),
    )
    font = float(pack.get("font_size", FONT))
    axis_font = float(pack.get("axis_label_size", AXIS_FONT))
    zoom = lerp(TRIO_ZOOM, FULL_ZOOM, smoothstep(focus))
    panel_font = lerp(font * 0.72, font, smoothstep(focus))
    panel_axis = lerp(axis_font * 0.72, axis_font, smoothstep(focus))
    for i, (rect, la, wb) in enumerate(zip(rects, alphas, lines)):
        if la < 0.22:
            continue

        ax = fig.add_axes(rect)
        child = dict(spec)
        child["data"] = "sep"
        child["w1"], child["w2"], child["b"] = wb
        child["drive"] = ""
        child["b_slide"] = 0.0
        child["xlabel_mode"] = "hours"
        child["x_scale"] = 1.0
        # Words only once the panel is large enough to hold them, except a light margin tag.
        if i < 2 and focus < 0.5:
            child["width_alpha"] = 0.0
        _draw_students(ax, pack, child, zoom=zoom, font=panel_font, axis_font=panel_axis)
        if la < 0.999:
            # Per-axes fade isn't available after raster; draw white by leaving it. Alpha is 0 or 1 mostly.
            pass
    img = fig_to_image(fig, dpi=int(pack.get("dpi", 120)))
    return _fade_image(img, float(spec.get("scene_alpha", 1.0) or 1.0))


def render_frame(pack: dict[str, Any], spec: dict[str, Any]):
    if spec.get("kind") == "trio":
        return _render_trio(pack, spec)
    return _render_stage(pack, spec)



def _vis(*on: int, n: int = 22) -> dict[str, float]:
    d = {f"v{i}": 0.0 for i in range(n)}
    for i in on:
        d[f"v{i}"] = 1.0
    return d


def _all_on() -> dict[str, float]:
    return _vis(*range(20))


def _alphas(**kw: float) -> dict[str, float]:
    d = {f"a{i}": 0.0 for i in (CORE_FAIL, CORE_PASS, MID_PT, LEFT_PT, RIGHT_PT, NOISE_FAIL, NOISE_PASS)}
    d.update({k: float(v) for k, v in kw.items()})
    return d


_TWO_X = np.array([2.0, 4.0])
_TWO_E = np.array([4.0, 2.0])
_TWO_Y = np.array([-1.0, 1.0])
_THIN = separator_at_angle(-2.0, _TWO_X, _TWO_E, _TWO_Y)
_MIDANG = separator_at_angle(-1.15, _TWO_X, _TWO_E, _TWO_Y)
_UNIT = separator_at_angle(-np.pi / 4.0, _TWO_X, _TWO_E, _TWO_Y)


def _end_01() -> dict[str, Any]:
    return _base(**_all_on(), line_style="solid", prob_alpha=0.0, point_alpha=1.0)


def _two_blank() -> dict[str, Any]:
    return _base(
        **_vis(CORE_FAIL, CORE_PASS),
        w1=PAIR_W[0], w2=PAIR_W[1], b=PAIR_B,
        line_alpha=0.0, strip=0.0,
    )


def _two_ready() -> dict[str, Any]:
    return _base(
        **_vis(CORE_FAIL, CORE_PASS),
        w1=PAIR_W[0], w2=PAIR_W[1], b=PAIR_B,
        strip=1.0, strip_mode="functional",
        label_alpha=1.0, width_alpha=1.0, perp_alpha=1.0, bisect_alpha=1.0,
    )


def _with_outside() -> dict[str, Any]:
    return _base(
        **_vis(CORE_FAIL, CORE_PASS, LEFT_PT, RIGHT_PT),
        **_alphas(**{f"a{CORE_FAIL}": PAIR_ALPHA, f"a{CORE_PASS}": PAIR_ALPHA}),
        w1=PAIR_W[0], w2=PAIR_W[1], b=PAIR_B,
        strip=1.0, strip_mode="functional",
        label_alpha=0.0, width_alpha=0.0, bisect_alpha=0.0,
        focus=-1, focus_alpha=0.0,
    )


def _moved() -> dict[str, Any]:
    return _base(
        **_vis(CORE_FAIL, CORE_PASS, LEFT_PT, RIGHT_PT, MID_PT),
        **_alphas(**{
            f"a{CORE_FAIL}": MOVED_ALPHA,
            f"a{CORE_PASS}": 0.0,
            f"a{MID_PT}": MOVED_ALPHA,
        }),
        w1=MOVED_W[0], w2=MOVED_W[1], b=MOVED_B,
        strip=1.0, strip_mode="functional",
        line2_alpha=0.65,
        w1b=PAIR_W[0], w2b=PAIR_W[1], bb=PAIR_B,
        inside_ring=0.0, focus=-1, focus_alpha=0.0,
    )


def plan_ch9_01(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Logistic probability field fades. The separating line and the full roster stay."""
    plan = _Plan(clip_id)
    hold = _N(20, 4)
    start = _base(**_all_on(), prob_alpha=1.0, line_style="auto", line_alpha=1.0, point_alpha=1.0)
    plan.hold(hold, "stage", **start)
    plan.tween(_N(48, 10), "stage", start, _end_01(), discrete=_DISC)
    plan.hold(hold, "stage", **_end_01())
    return plan.specs


def plan_ch9_02(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Set students aside until only one pass and one fail remain."""
    plan = _Plan(clip_id)
    hold = _N(10, 2)
    start = _end_01()
    # Far from the gap, then the near-margin crowd, then the line itself leaves.
    wave1 = (8, 7, 19, 18, 14, 15, 13, 3)
    wave2 = (6, 5, 2, 16, 17, 11, 1)
    wave3 = (0, 9, 10)
    visible = set(range(20))
    plan.hold(hold, "stage", **start)
    prev = start
    for wave in (wave1, wave2, wave3):
        visible -= set(wave)
        nxt = dict(prev)
        nxt.update(_vis(*visible))
        if wave is wave1:
            nxt["line_alpha"] = 0.0
            nxt["prob_alpha"] = 0.0
        plan.tween(_N(28, 6), "stage", prev, nxt, discrete=_DISC)
        plan.hold(_N(8, 2), "stage", **nxt)
        prev = nxt
    blank = _two_blank()
    plan.tween(_N(8, 2), "stage", prev, blank, discrete=_DISC)
    plan.hold(_N(12, 3), "stage", **blank)
    return plan.specs


def plan_ch9_03(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """For two students the best line is the perpendicular bisector, centered in the gap."""
    plan = _Plan(clip_id)
    hold = _N(10, 2)
    blank = _two_blank()
    thin = _base(
        **_vis(CORE_FAIL, CORE_PASS),
        w1=_THIN[0], w2=_THIN[1], b=_THIN[2],
        line_alpha=1.0, strip=1.0, strip_mode="geom",
    )
    mid = _base(
        **_vis(CORE_FAIL, CORE_PASS),
        w1=_MIDANG[0], w2=_MIDANG[1], b=_MIDANG[2],
        line_alpha=1.0, strip=1.0, strip_mode="geom",
    )
    unit = _base(
        **_vis(CORE_FAIL, CORE_PASS),
        w1=_UNIT[0], w2=_UNIT[1], b=_UNIT[2],
        line_alpha=1.0, strip=1.0, strip_mode="geom",
    )
    off = dict(unit)
    off.update(b=-0.55, both_perp=1.0, label_alpha=1.0)
    centered = dict(unit)
    centered.update(b=0.0, both_perp=1.0, label_alpha=1.0, width_alpha=1.0, bisect_alpha=1.0)
    ready = _two_ready()
    plan.hold(_N(8, 2), "stage", **blank)
    plan.tween(_N(18, 4), "stage", blank, thin, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **thin)
    plan.tween(_N(22, 5), "stage", thin, mid, discrete=_DISC)
    plan.tween(_N(22, 5), "stage", mid, unit, discrete=_DISC)
    plan.hold(_N(6, 2), "stage", **unit)
    plan.tween(_N(16, 4), "stage", unit, off, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **off)
    plan.tween(_N(24, 6), "stage", off, centered, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **centered)
    plan.tween(_N(16, 4), "stage", centered, ready, discrete=_DISC)
    plan.hold(hold, "stage", **ready)
    return plan.specs


def plan_ch9_04(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """A student further to either side does not move the boundary."""
    plan = _Plan(clip_id)
    hold = _N(12, 3)
    start = _two_ready()
    left = dict(start)
    left.update(
        _vis(CORE_FAIL, CORE_PASS, LEFT_PT),
        focus=LEFT_PT, focus_alpha=1.0, bisect_alpha=0.15,
        label_alpha=0.0, width_alpha=0.0,
    )
    right = dict(left)
    right.update(
        _vis(CORE_FAIL, CORE_PASS, LEFT_PT, RIGHT_PT),
        focus=RIGHT_PT, focus_alpha=1.0, bisect_alpha=0.0, label_alpha=0.0, width_alpha=0.0,
    )
    done = _with_outside()
    plan.hold(_N(8, 2), "stage", **start)
    plan.tween(_N(20, 5), "stage", start, left, discrete=_DISC)
    plan.hold(_N(12, 3), "stage", **left)
    plan.tween(_N(20, 5), "stage", left, right, discrete=_DISC)
    plan.hold(_N(12, 3), "stage", **right)
    plan.tween(_N(12, 3), "stage", right, done, discrete=_DISC)
    plan.hold(hold, "stage", **done)
    return plan.specs


def plan_ch9_05(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """A student between the two edges forces a new boundary."""
    plan = _Plan(clip_id)
    hold = _N(12, 3)
    start = _with_outside()
    arrived = dict(start)
    arrived.update(
        **_vis(CORE_FAIL, CORE_PASS, LEFT_PT, RIGHT_PT, MID_PT),
        focus=MID_PT, focus_alpha=1.0, inside_ring=1.0,
    )
    moved = _moved()
    moved.update(focus=MID_PT, focus_alpha=0.35, inside_ring=1.0)
    quiet = _moved()
    plan.hold(_N(8, 2), "stage", **start)
    plan.tween(_N(18, 4), "stage", start, arrived, discrete=_DISC)
    plan.hold(_N(12, 3), "stage", **arrived)
    plan.tween(_N(36, 8), "stage", arrived, moved, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **moved)
    plan.tween(_N(12, 3), "stage", moved, quiet, discrete=_DISC)
    plan.hold(hold, "stage", **quiet)
    return plan.specs


def plan_ch9_06(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Return to the two-student line. Scaling the knobs changes scores, not distances."""
    plan = _Plan(clip_id)
    hold = _N(8, 2)
    start = _moved()
    back = _with_outside()
    scores = dict(back)
    scores.update(score_alpha=1.0, score_gain=1.0, strip=1.0)
    big = dict(scores)
    big.update(w1=2.0, w2=-2.0, b=0.0, strip=4.0, score_gain=4.0)
    walked = dict(big)
    walked.update(walk=1.0, score_alpha=0.45)
    home = dict(back)
    plan.hold(_N(8, 2), "stage", **start)
    plan.tween(_N(28, 6), "stage", start, back, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **back)
    plan.tween(_N(14, 4), "stage", back, scores, discrete=_DISC)
    plan.tween(_N(22, 5), "stage", scores, big, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **big)
    plan.tween(_N(16, 4), "stage", big, walked, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **walked)
    plan.tween(_N(22, 5), "stage", walked, home, discrete=_DISC)
    plan.hold(hold, "stage", **home)
    return plan.specs


def plan_ch9_07(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Tour the allowed wedge, settle on its lowest contour, then step past the boundary."""
    plan = _Plan(clip_id)
    hold = _N(8, 2)
    start = _with_outside()
    opened = dict(start)
    opened.update(split=1.0, right="slice", right_alpha=1.0, cons_n=0.0, eq_alpha=1.0)
    wedged = dict(opened)
    wedged.update(cons_n=2.0)
    # Interior of the white wedge: the strip is narrower, both students still outside it.
    deep = dict(wedged)
    deep.update(w1=1.0, w2=-1.0, b=0.0)
    side = dict(deep)
    side.update(w1=0.85, w2=-0.95, b=0.0)
    best = dict(wedged)
    best.update(w1=PAIR_W[0], w2=PAIR_W[1], b=PAIR_B, inside_ring=0.0)
    beyond = dict(best)
    beyond.update(w1=0.16, w2=-0.16, b=0.0, inside_ring=1.0)
    back = dict(best)
    back.update(inside_ring=0.0)
    plan.hold(_N(6, 2), "stage", **start)
    plan.tween(_N(14, 4), "stage", start, opened, discrete=_DISC)
    plan.hold(_N(6, 2), "stage", **opened)
    plan.tween(_N(16, 4), "stage", opened, wedged, discrete=_DISC)
    plan.hold(_N(4, 2), "stage", **wedged)
    plan.tween(_N(20, 5), "stage", wedged, deep, discrete=_DISC)
    plan.tween(_N(16, 4), "stage", deep, side, discrete=_DISC)
    plan.hold(_N(10, 2), "stage", **side)
    plan.tween(_N(20, 5), "stage", side, best, discrete=_DISC)
    plan.hold(_N(14, 3), "stage", **best)
    plan.tween(_N(20, 5), "stage", best, beyond, discrete=_DISC)
    plan.hold(_N(10, 2), "stage", **beyond)
    plan.tween(_N(16, 4), "stage", beyond, back, discrete=_DISC)
    plan.hold(hold, "stage", **back)
    return plan.specs


def plan_ch9_08(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """The same objective as a voxel lattice. b is free in the color, pinned by the constraints."""
    plan = _Plan(clip_id)
    hold = _N(8, 2)
    start = dict(_with_outside())
    start.update(split=1.0, right="slice", right_alpha=1.0, cons_n=2.0, eq_alpha=1.0)
    blank = dict(start)
    blank.update(right_alpha=0.0)
    hidden = dict(blank)
    hidden.update(right="voxel", right_alpha=0.0, sweep=0.0, carve=0.0, marker_alpha=1.0, cam_u=0.0, vox_grow=0.0)
    voxels = dict(hidden)
    voxels.update(right_alpha=1.0, vox_grow=1.0)
    swept = dict(voxels)
    swept.update(sweep=1.0, cam_u=0.4)
    carved = dict(swept)
    carved.update(carve=1.0, cam_u=0.75)
    longer = dict(carved)
    longer.update(w1=1.0, w2=-1.0, b=0.0, eq_alpha=1.0)
    slid = dict(longer)
    slid.update(b=0.7)
    back_b = dict(longer)
    solved = dict(carved)
    solved.update(w1=PAIR_W[0], w2=PAIR_W[1], b=PAIR_B, cam_u=1.0, ghost=0.0)
    ghost = dict(solved)
    ghost.update(ghost=1.0)
    plan.hold(_N(6, 2), "stage", **start)
    plan.tween(_N(8, 2), "stage", start, blank, discrete=_DISC)
    plan.tween(_N(4, 2), "stage", blank, hidden, discrete=_DISC)
    plan.tween(_N(14, 3), "stage", hidden, voxels, discrete=_DISC)
    plan.tween(_N(26, 6), "stage", voxels, swept, discrete=_DISC)
    plan.tween(_N(22, 5), "stage", swept, carved, discrete=_DISC)
    plan.hold(_N(6, 2), "stage", **carved)
    plan.tween(_N(18, 4), "stage", carved, longer, discrete=_DISC)
    plan.hold(_N(6, 2), "stage", **longer)
    plan.tween(_N(16, 4), "stage", longer, slid, discrete=_DISC)
    plan.tween(_N(16, 4), "stage", slid, back_b, discrete=_DISC)
    plan.tween(_N(22, 5), "stage", back_b, solved, discrete=_DISC)
    plan.hold(_N(6, 2), "stage", **solved)
    plan.tween(_N(12, 3), "stage", solved, ghost, discrete=_DISC)
    plan.tween(_N(12, 3), "stage", ghost, solved, discrete=_DISC)
    plan.hold(hold, "stage", **solved)
    return plan.specs


def plan_ch9_09(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Lagrangian terms, then the weight vector assembled from the two students."""
    plan = _Plan(clip_id)
    hold = _N(8, 2)
    start = dict(_with_outside())
    start.update(
        split=1.0, right="voxel", right_alpha=1.0, sweep=1.0, carve=1.0,
        marker_alpha=1.0, cam_u=1.0, eq_alpha=1.0, cons_n=2.0, vox_grow=1.0,
    )
    blank = dict(start)
    blank.update(right_alpha=0.0, **_alphas())
    tucked = dict(blank)
    tucked.update(right="vectors", right_alpha=0.0, vox_grow=0.0, lag_step=0.0, eq_alpha=0.0)
    quiet = _with_outside()
    quiet.update(
        split=1.0, right="vectors", right_alpha=1.0, lag_step=0.0,
        **_alphas(),
        alpha_label=0.0, vox_grow=0.0,
    )
    words = dict(quiet)
    words.update(lag_step=3.0)
    grown = dict(words)
    grown.update(
        **_alphas(**{f"a{CORE_FAIL}": PAIR_ALPHA, f"a{CORE_PASS}": PAIR_ALPHA}),
        alpha_label=1.0,
    )
    plan.hold(_N(6, 2), "stage", **start)
    plan.tween(_N(8, 2), "stage", start, blank, discrete=_DISC)
    plan.tween(_N(4, 2), "stage", blank, tucked, discrete=_DISC)
    plan.tween(_N(12, 3), "stage", tucked, quiet, discrete=_DISC)
    plan.tween(_N(28, 6), "stage", quiet, words, discrete=_DISC)
    plan.hold(_N(6, 2), "stage", **words)
    plan.tween(_N(28, 6), "stage", words, grown, discrete=_DISC)
    plan.hold(hold, "stage", **grown)
    return plan.specs


def plan_ch9_10(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """One shared α. The bowl is the weight size. Left of the wall the strip covers the students."""
    plan = _Plan(clip_id)
    hold = _N(10, 2)
    start = dict(_with_outside())
    start.update(
        split=1.0, right="vectors", right_alpha=1.0, lag_step=3.0, alpha_label=1.0,
        **_alphas(**{f"a{CORE_FAIL}": PAIR_ALPHA, f"a{CORE_PASS}": PAIR_ALPHA}),
    )

    def at(t: float, **kw: Any) -> dict[str, Any]:
        d = dict(start)
        tt = float(t)
        d.update(
            right="dual", dual_kind="pair", right_alpha=1.0, lag_step=0.0,
            quad_alpha=1.0, formula_step=1.0, primal_alpha=0.0, bound_t=tt,
            w1=2.0 * tt, w2=-2.0 * tt, b=0.0,
            **_alphas(**{f"a{CORE_FAIL}": tt, f"a{CORE_PASS}": tt}),
            alpha_label=1.0, inside_ring=1.0 if tt < PAIR_ALPHA - 0.02 else 0.0,
            caption="", caption_alpha=0.0,
        )
        d.update(kw)
        return d

    blank = dict(start)
    blank.update(right_alpha=0.0, lag_step=0.0)
    hidden = at(PAIR_ALPHA, right_alpha=0.0, formula_step=0.0, quad_alpha=0.0)
    shown = at(PAIR_ALPHA)
    high = at(0.48)
    walled = at(0.48, formula_step=2.0)
    touch = at(PAIR_ALPHA, formula_step=2.0)
    low = at(0.08, formula_step=2.0, inside_ring=1.0)
    back = at(PAIR_ALPHA, formula_step=2.0, inside_ring=0.0)
    best = at(
        PAIR_ALPHA, formula_step=3.0, primal_alpha=1.0, quad_alpha=1.0, inside_ring=0.0,
    )
    plan.hold(_N(6, 2), "stage", **start)
    plan.tween(_N(8, 2), "stage", start, blank, discrete=_DISC)
    plan.tween(_N(4, 2), "stage", blank, hidden, discrete=_DISC)
    plan.tween(_N(16, 4), "stage", hidden, shown, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **shown)
    plan.tween(_N(22, 5), "stage", shown, high, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **high)
    plan.tween(_N(12, 3), "stage", high, walled, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **walled)
    plan.tween(_N(16, 4), "stage", walled, touch, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **touch)
    plan.tween(_N(18, 4), "stage", touch, low, discrete=_DISC)
    plan.hold(_N(10, 2), "stage", **low)
    plan.tween(_N(16, 4), "stage", low, back, discrete=_DISC)
    plan.hold(_N(10, 2), "stage", **back)
    plan.tween(_N(16, 4), "stage", back, best, discrete=_DISC)
    plan.hold(hold, "stage", **best)
    return plan.specs


def plan_ch9_11(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """A student inside the gap rules out the old line. The slice, the landscape, and α all move."""
    plan = _Plan(clip_id)
    hold = _N(10, 2)
    start = dict(_with_outside())
    start.update(
        split=1.0, right="dual", dual_kind="pair", right_alpha=1.0,
        quad_alpha=1.0, primal_alpha=1.0, formula_step=3.0, bound_t=PAIR_ALPHA,
        alpha_label=1.0,
        **_alphas(**{f"a{CORE_FAIL}": PAIR_ALPHA, f"a{CORE_PASS}": PAIR_ALPHA}),
    )
    arrived = dict(start)
    arrived.update(
        **_vis(CORE_FAIL, CORE_PASS, LEFT_PT, RIGHT_PT, MID_PT),
        **_alphas(**{f"a{CORE_FAIL}": PAIR_ALPHA, f"a{CORE_PASS}": PAIR_ALPHA, f"a{MID_PT}": 0.0}),
        focus=MID_PT, focus_alpha=1.0, inside_ring=1.0,
        caption="sits in the gap", caption_alpha=1.0,
    )
    blank = dict(arrived)
    blank.update(right_alpha=0.0)
    slice_hidden = dict(blank)
    slice_hidden.update(right="slice", right_alpha=0.0, cons_n=2.0, extra=0.0, eq_alpha=1.0)
    sliced = dict(slice_hidden)
    sliced.update(right_alpha=1.0)
    cut = dict(sliced)
    cut.update(cons_n=3.0, extra=1.0, caption="old line no longer allowed", caption_alpha=1.0)
    solved = dict(cut)
    solved.update(
        w1=MOVED_W[0], w2=MOVED_W[1], b=MOVED_B,
        **_alphas(**{f"a{CORE_FAIL}": MOVED_ALPHA, f"a{CORE_PASS}": 0.0, f"a{MID_PT}": MOVED_ALPHA}),
        inside_ring=0.0, focus_alpha=0.55, line2_alpha=0.65,
        caption="", caption_alpha=0.0,
    )
    vox_off = dict(solved)
    vox_off.update(right_alpha=0.0)
    vox_swap = dict(vox_off)
    vox_swap.update(
        right="voxel", right_alpha=0.0, vox_grow=0.0,
        sweep=1.0, carve=1.0, marker_alpha=1.0, cam_u=0.2, extra=1.0,
    )
    vox_on = dict(vox_swap)
    vox_on.update(right_alpha=1.0, vox_grow=1.0, cam_u=1.0)
    dual_off = dict(vox_on)
    dual_off.update(right_alpha=0.0)
    dual_swap = dict(dual_off)
    dual_swap.update(
        right="dual", dual_kind="moved", right_alpha=0.0, vox_grow=0.0,
        bound_t=MOVED_ALPHA, dual_ghost=1.0, primal_alpha=0.0,
        quad_alpha=1.0, formula_step=1.0,
    )
    dual_bowl = dict(dual_swap)
    dual_bowl.update(right_alpha=1.0)
    dual_wall = dict(dual_bowl)
    dual_wall.update(formula_step=2.0)
    dual_best = dict(dual_wall)
    dual_best.update(
        formula_step=3.0, primal_alpha=1.0,
        caption="narrower gap, larger α", caption_alpha=1.0,
    )
    vec_off = dict(dual_best)
    vec_off.update(right_alpha=0.0, caption="", caption_alpha=0.0)
    vec_swap = dict(vec_off)
    vec_swap.update(right="vectors", right_alpha=0.0, lag_step=0.0)
    vec = dict(vec_swap)
    vec.update(right_alpha=1.0, alpha_label=1.0)
    vec_words = dict(vec)
    vec_words.update(lag_step=3.0)
    closed = _moved()
    closed.update(
        alpha_label=1.0, sv_pulse=0.0, split=0.0, right="none", right_alpha=1.0,
        caption="", caption_alpha=0.0, vox_grow=0.0,
    )
    plan.hold(_N(6, 2), "stage", **start)
    plan.tween(_N(16, 4), "stage", start, arrived, discrete=_DISC)
    plan.hold(_N(12, 3), "stage", **arrived)
    plan.tween(_N(8, 2), "stage", arrived, blank, discrete=_DISC)
    plan.tween(_N(4, 2), "stage", blank, slice_hidden, discrete=_DISC)
    plan.tween(_N(12, 3), "stage", slice_hidden, sliced, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **sliced)
    plan.tween(_N(18, 4), "stage", sliced, cut, discrete=_DISC)
    plan.hold(_N(10, 2), "stage", **cut)
    plan.tween(_N(28, 6), "stage", cut, solved, discrete=_DISC)
    plan.hold(_N(10, 2), "stage", **solved)
    plan.tween(_N(8, 2), "stage", solved, vox_off, discrete=_DISC)
    plan.tween(_N(4, 2), "stage", vox_off, vox_swap, discrete=_DISC)
    plan.tween(_N(16, 4), "stage", vox_swap, vox_on, discrete=_DISC)
    plan.hold(_N(12, 3), "stage", **vox_on)
    plan.tween(_N(8, 2), "stage", vox_on, dual_off, discrete=_DISC)
    plan.tween(_N(4, 2), "stage", dual_off, dual_swap, discrete=_DISC)
    plan.tween(_N(14, 3), "stage", dual_swap, dual_bowl, discrete=_DISC)
    plan.hold(_N(6, 2), "stage", **dual_bowl)
    plan.tween(_N(12, 3), "stage", dual_bowl, dual_wall, discrete=_DISC)
    plan.hold(_N(6, 2), "stage", **dual_wall)
    plan.tween(_N(14, 3), "stage", dual_wall, dual_best, discrete=_DISC)
    plan.hold(_N(10, 2), "stage", **dual_best)
    plan.tween(_N(8, 2), "stage", dual_best, vec_off, discrete=_DISC)
    plan.tween(_N(4, 2), "stage", vec_off, vec_swap, discrete=_DISC)
    plan.tween(_N(12, 3), "stage", vec_swap, vec, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **vec)
    plan.tween(_N(14, 3), "stage", vec, vec_words, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **vec_words)
    plan.tween(_N(16, 4), "stage", vec_words, closed, discrete=_DISC)
    plan.hold(hold, "stage", **closed)
    return plan.specs


def plan_ch9_12(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Two crossing students return. Slack and a price C replace a hard gap."""
    plan = _Plan(clip_id)
    hold = _N(8, 2)
    Cs, wb = small_soft_table()
    start = _moved()
    start.update(alpha_label=1.0, line2_alpha=0.65)
    clear = dict(start)
    clear.update(line2_alpha=0.0, alpha_label=0.0)
    crossed = dict(clear)
    crossed.update(
        **_vis(CORE_FAIL, CORE_PASS, LEFT_PT, RIGHT_PT, MID_PT, NOISE_FAIL, NOISE_PASS),
        wrong_ring=1.0, inside_ring=1.0,
    )

    def _at(i: int, u: float) -> dict[str, Any]:
        d = dict(crossed)
        d.update(
            w1=float(wb[i, 0]), w2=float(wb[i, 1]), b=float(wb[i, 2]),
            c_value=float(Cs[i]), slider_u=u, slider_alpha=1.0, slack_alpha=1.0,
            strip=1.0, strip_mode="functional",
        )
        return d

    soft0 = _at(0, 0.0)
    plan.hold(_N(6, 2), "stage", **start)
    plan.tween(_N(12, 3), "stage", start, clear, discrete=_DISC)
    plan.tween(_N(16, 4), "stage", clear, crossed, discrete=_DISC)
    plan.hold(_N(10, 2), "stage", **crossed)
    plan.tween(_N(20, 5), "stage", crossed, soft0, discrete=_DISC)
    prev = soft0
    for i in range(1, len(Cs)):
        nxt = _at(i, i / (len(Cs) - 1))
        plan.tween(_N(16, 4), "stage", prev, nxt, discrete=_DISC)
        plan.hold(_N(6, 2), "stage", **nxt)
        prev = nxt
    spot = dict(prev)
    spot.update(spot=NOISE_FAIL, spot_alpha=1.0, slider_alpha=0.4)
    hinged = dict(spot)
    hinged.update(
        split=1.0, right="loss", right_alpha=1.0, hinge_alpha=1.0, logit_alpha=0.0,
        glow=0.85, glow_mode="hinge", spot=-1, spot_alpha=0.0,
    )
    plan.tween(_N(12, 3), "stage", prev, spot, discrete=_DISC)
    plan.hold(_N(10, 2), "stage", **spot)
    plan.tween(_N(18, 4), "stage", spot, hinged, discrete=_DISC)
    plan.hold(hold, "stage", **hinged)
    return plan.specs


def plan_ch9_13(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Hinge loss stops at the margin. Logistic loss keeps charging, and its line is different."""
    plan = _Plan(clip_id)
    hold = _N(10, 2)
    # Rebuild the soft-margin end so this clip opens where the previous one closed.
    Cs, wb = small_soft_table()
    i = len(Cs) - 1
    lw = small_logistic_fit()
    start = _base(
        **_vis(CORE_FAIL, CORE_PASS, LEFT_PT, RIGHT_PT, MID_PT, NOISE_FAIL, NOISE_PASS),
        w1=float(wb[i, 0]), w2=float(wb[i, 1]), b=float(wb[i, 2]),
        strip=1.0, strip_mode="functional",
        wrong_ring=1.0, inside_ring=1.0,
        slider_alpha=0.4, slider_u=1.0, c_value=float(Cs[i]), slack_alpha=1.0,
        split=1.0, right="loss", right_alpha=1.0, hinge_alpha=1.0, logit_alpha=0.0,
        glow=0.85, glow_mode="hinge",
    )
    both = dict(start)
    both.update(logit_alpha=1.0)
    switched = dict(both)
    switched.update(glow_mode="logistic")
    lined = dict(switched)
    lined.update(
        line2_alpha=1.0, line2_color=LOGIT_COLOR,
        w1b=lw[0], w2b=lw[1], bb=lw[2],
    )
    plan.hold(hold, "stage", **start)
    plan.tween(_N(20, 5), "stage", start, both, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **both)
    plan.tween(_N(16, 4), "stage", both, switched, discrete=_DISC)
    plan.hold(_N(8, 2), "stage", **switched)
    plan.tween(_N(18, 4), "stage", switched, lined, discrete=_DISC)
    plan.hold(hold, "stage", **lined)
    return plan.specs


def plan_ch9_14(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """A straight line cannot follow a bent roster."""
    plan = _Plan(clip_id)
    hold = _N(14, 3)
    Cs, wb = small_soft_table()
    i = len(Cs) - 1
    lw = small_logistic_fit()
    start = _base(
        **_vis(CORE_FAIL, CORE_PASS, LEFT_PT, RIGHT_PT, MID_PT, NOISE_FAIL, NOISE_PASS),
        w1=float(wb[i, 0]), w2=float(wb[i, 1]), b=float(wb[i, 2]),
        strip=1.0, strip_mode="functional",
        wrong_ring=1.0, inside_ring=1.0,
        slider_alpha=0.4, slider_u=1.0, c_value=float(Cs[i]), slack_alpha=1.0,
        split=1.0, right="loss", right_alpha=1.0, hinge_alpha=1.0, logit_alpha=1.0,
        glow=0.85, glow_mode="logistic",
        line2_alpha=1.0, line2_color=LOGIT_COLOR, w1b=lw[0], w2b=lw[1], bb=lw[2],
    )
    single = dict(start)
    single.update(split=0.0, right="none", right_alpha=0.0, glow=0.0, slider_alpha=0.0, slack_alpha=0.0)
    dark = dict(single)
    dark.update(scene_alpha=0.0)
    moons = _base(
        data="curve", scene_alpha=0.0,
        w1=1.0, w2=-0.15, b=-2.2,
        strip=0.0, line_alpha=1.0, wrong_ring=1.0,
        line2_alpha=0.0, glow=0.0, slider_alpha=0.0,
    )
    show = dict(moons)
    show.update(scene_alpha=1.0)
    plan.hold(_N(6, 2), "stage", **start)
    plan.tween(_N(14, 4), "stage", start, single, discrete=_DISC)
    plan.tween(_N(12, 3), "stage", single, dark, discrete=_DISC)
    plan.hold(_N(3, 2), "stage", **moons)
    plan.tween(_N(12, 3), "stage", moons, show, discrete=_DISC)
    prev = show
    for ang, bb in ((-0.25, -2.381), (1.20, -4.272), (0.55, -4.538), (-np.pi / 4.0, 0.0)):
        nxt = dict(show)
        nxt.update(w1=float(np.cos(ang)), w2=float(np.sin(ang)), b=float(bb), wrong_ring=1.0, line_alpha=1.0)
        plan.tween(_N(16, 4), "stage", prev, nxt, discrete=_DISC)
        plan.hold(_N(6, 2), "stage", **nxt)
        prev = nxt
    plan.hold(hold, "stage", **prev)
    return plan.specs


__all__ = [
    "build_pack_ch9",
    "render_frame",
    "plan_ch9_01",
    "plan_ch9_02",
    "plan_ch9_03",
    "plan_ch9_04",
    "plan_ch9_05",
    "plan_ch9_06",
    "plan_ch9_07",
    "plan_ch9_08",
    "plan_ch9_09",
    "plan_ch9_10",
    "plan_ch9_11",
    "plan_ch9_12",
    "plan_ch9_13",
    "plan_ch9_14",
]
